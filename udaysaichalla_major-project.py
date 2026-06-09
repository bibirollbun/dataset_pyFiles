#!python --version


!conda create -n myenv python=3.8 -y


#!pip install typing-extensions


!pip install --upgrade transformers timm



"""try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal"""


!pip install --upgrade transformers



"""try:
    from transformers import ViTForImageClassification, ViTFeatureExtractor
except ModuleNotFoundError:
    !pip install transformers
    from transformers import ViTForImageClassification, ViTFeatureExtractor
import random"""


#!pip install --upgrade transformers

# Gerekli kütüphaneleri yükleyelim
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
import random
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns



import os 
# Metadata ve veri yolları
metadata_path = '/kaggle/input/dfdc-48/dfdc_train_part_48/metadata.json'
input_dir = '/kaggle/input/dfdc-48/dfdc_train_part_48'

# Ana veri klasörleri
base_dir = '/kaggle/working/dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Alt klasörler için fonksiyon
def create_split_subdirs(split_dir):
    real_dir = os.path.join(split_dir, 'real')
    fake_dir = os.path.join(split_dir, 'fake')
    real_frame_dir = os.path.join(split_dir, 'real_frame')
    fake_frame_dir = os.path.join(split_dir, 'fake_frame')
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_frame_dir, exist_ok=True)
    os.makedirs(fake_frame_dir, exist_ok=True)
    
    return real_dir, fake_dir, real_frame_dir, fake_frame_dir


# Tüm klasörleri oluştur
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)
    create_split_subdirs(split_dir)

# Metadata dosyasını yükle
import json
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Videoların bulunduğu input klasörü
input_dir = '/kaggle/input/dfdc-48/dfdc_train_part_48'

# Video Düzeyinde Veri Bölme
import random
from sklearn.model_selection import train_test_split


def split_videos(sample_size=10):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    real_videos, fake_videos = [], []

    for video_name, data in metadata.items():
        video_path = os.path.join(input_dir, video_name)
        if os.path.exists(video_path):
            if data['label'] == 'REAL':
                real_videos.append((video_path, 'real'))
            elif data['label'] == 'FAKE':
                fake_videos.append((video_path, 'fake'))

    real_videos = random.sample(real_videos, min(len(real_videos), sample_size))
    fake_videos = random.sample(fake_videos, min(len(fake_videos), sample_size))

    all_videos = real_videos + fake_videos
    labels = [1 if v[1] == 'real' else 0 for v in all_videos]

    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos, labels, test_size=0.3, stratify=labels, random_state=42)
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
    
    return train_videos, val_videos, test_videos



# Videoları kopyalama fonksiyonu
import shutil
def copy_videos_to_split_dirs(videos, split_dir):
    real_dir, fake_dir, _, _ = create_split_subdirs(split_dir)
    
    for video_path, label in videos:
        target_dir = real_dir if label == 'real' else fake_dir
        video_name = os.path.basename(video_path)
        shutil.copy(video_path, os.path.join(target_dir, video_name))
        print(f"Copied {video_name} to {target_dir}")

train_videos, val_videos, test_videos = split_videos()
copy_videos_to_split_dirs(train_videos, train_dir)
copy_videos_to_split_dirs(val_videos, val_dir)
copy_videos_to_split_dirs(test_videos, test_dir)


# Videodan yüzleri çıkarmak ve kare atlama ile çerçeve işleme süresini azaltmak
def extract_faces_from_videos(split_dir, frame_skip=10):
    real_dir, fake_dir, real_frame_dir, fake_frame_dir = create_split_subdirs(split_dir)
    
    # Gerçek videolardan frame çıkar
    for video_name in os.listdir(real_dir):
        video_path = os.path.join(real_dir, video_name)
        if os.path.isfile(video_path):
            extract_faces_from_video(video_path, real_frame_dir, frame_skip)
    
    # Sahte videolardan frame çıkar
    for video_name in os.listdir(fake_dir):
        video_path = os.path.join(fake_dir, video_name)
        if os.path.isfile(video_path):
            extract_faces_from_video(video_path, fake_frame_dir, frame_skip)


"""
!pip install mtcnn
!pip install facenet-pytorch==2.5.0
!pip install torch torchvision torchaudio
from facenet_pytorch import MTCNN
# MTCNN yüz algılama modeli
mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')

def extract_faces_from_video(video_path, output_dir, frame_skip):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        # Frame'i işle
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized_frame = cv2.equalizeHist(gray_frame)
        frame_rgb = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                try:
                    face_resized = cv2.resize(face, (224, 224))
                    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame{frame_count}_face{i}.jpg")
                    cv2.imwrite(output_path, face_resized)
                except cv2.error as e:
                    print(f"[Resize Error] Video: {os.path.basename(video_path)}, Frame: {frame_count}, Face: {i}, Error: {e}")
                    continue
    
    cap.release()
    print(f"Processed {os.path.basename(video_path)}")"""


#OPENCV

# Yüz çıkarma fonksiyonu
def extract_faces_from_video(video_path, output_dir, frame_skip=10):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        for i, (x, y, w, h) in enumerate(faces):
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame{frame_count}_face{i}.jpg")
            cv2.imwrite(output_path, face_resized)
            face_count += 1
    cap.release()
    print(f"{video_path}: {face_count} faces extracted.")



 # 3. Her set için frame çıkar
    print("\nExtracting frames from training set...")
    extract_faces_from_videos(train_dir)
    print("\nExtracting frames from validation set...")
    extract_faces_from_videos(val_dir)
    print("\nExtracting frames from test set...")
    extract_faces_from_videos(test_dir)


from PIL import Image
import os
# Görüntüleri normalize etme
def normalize_images(input_dir, output_dir, image_size=224):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in os.listdir(input_dir):
        try:
            img_path = os.path.join(input_dir, img_name)
            img = Image.open(img_path).convert('RGB')  # Renk formatını RGB'ye dönüştür
            img = img.resize((image_size, image_size))  # Boyutlandır
            img.save(os.path.join(output_dir, img_name))  # Normalleştirilmiş görüntüyü kaydet
        except Exception as e:
            print(f"Hata oluştu: {img_name} - {e}")


normalize_images(train_dir+"/fake_frame","norm_train_dir/fake")
normalize_images(train_dir+"/real_frame","norm_train_dir/real")
normalize_images(test_dir+"/fake_frame","norm_test_dir/fake")
normalize_images(test_dir+"/real_frame","norm_test_dir/real")
normalize_images(val_dir+"/fake_frame","norm_val_dir/fake")
normalize_images(val_dir+"/real_frame","norm_val_dir/real")



# Veri Dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, norm_dir, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        
        real_dir = os.path.join(norm_dir, 'real')
        fake_dir = os.path.join(norm_dir, 'fake')
        
        # Real verileri oku
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                img_path = os.path.join(real_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(0)  
        
        # Fake verileri oku
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                img_path = os.path.join(fake_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(img_path)
                    self.labels.append(1)  

        # Veri seti boşsa uyarı ver
        if len(self.data) == 0:
            print(f"Uyarı: {norm_dir} dizininde veri bulunamadı!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, label

# ViT için veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#Dataset'leri tekrar oluştur ve veri kontrolü yap
train_dataset = FaceDataset("norm_train_dir", transform=transform)
test_dataset = FaceDataset("norm_test_dir", transform=transform)
val_dataset = FaceDataset("norm_val_dir", transform=transform)

#Veri kontrolü ekle
print(f"Train veri sayısı: {len(train_dataset)}")
print(f"Test veri sayısı: {len(test_dataset)}")
print(f"Validation veri sayısı: {len(val_dataset)}")

#Eğer veri yoksa hata verecektir
if len(train_dataset) == 0 or len(test_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Veri seti boş. Lütfen veri yollarını kontrol edin.")

# DataLoader'ları oluştur
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

print("Veri yükleyiciler başarıyla oluşturuldu.")



# Model Yükleme ve Eğitimi
import timm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
vit_model.head = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(in_features=768, out_features=2)
)
vit_model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Modeli eğitir ve validasyon performansını izler.
    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        criterion: Kayıp fonksiyonu
        optimizer: Optimizer
        num_epochs: Epoch sayısı
    """
    # Metrik takibi için listeler
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Eğitim Modu
        model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Sıfırlama
            optimizer.zero_grad()
            
            # Öngörü
            outputs = model(images)
            
            # Kaybı hesapla
            loss = criterion(outputs, labels)
            
            # Geriye yayılım
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Doğru tahmin sayısını hesapla
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Eğitim kaybı ve doğruluğu
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        
        # Metrikleri listelere ekle
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Doğrulama Performansı
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    # Eğitim geçmişini döndür
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }
    
    return history

def evaluate_model(model, loader, criterion):
    """
    Modeli değerlendirir ve kayıp ile doğruluk değerlerini döndürür.
    Args:
        model: Değerlendirilecek model
        loader: Değerlendirme için veri yükleyicisi
        criterion: Kayıp fonksiyonu
    Returns:
        avg_loss: Ortalama kayıp
        accuracy: Doğruluk oranı
    """
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Öngörü
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Doğru tahmin sayısını hesapla
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


# Test için yardımcı fonksiyon
def test_model(model, test_loader, criterion):
    """
    Modeli test seti üzerinde değerlendirir.
    Args:
        model: Test edilecek model
        test_loader: Test veri yükleyicisi
        criterion: Kayıp fonksiyonu
    Returns:
        test_loss: Test kaybı
        test_accuracy: Test doğruluğu
    """
    print("\nTest seti üzerinde değerlendiriliyor...")
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy
    
def evaluate_train_data(model, train_loader):
    model.eval()  # Modeli eval moduna al
    correct_predictions = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Öngörü
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Doğru tahmin sayısını hesapla
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    # Ortalama eğitim kaybı ve doğruluğu
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")


# Modeli kaydet
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model {file_path} konumuna kaydedildi.")

# Modeli yükle
def load_model(file_path, model):
    model.load_state_dict(torch.load(file_path))
    model = model.to(device)
    model.eval()
    print(f"Model {file_path} konumundan yüklendi.")
    return model


# Learning Rate Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ReduceLROnPlateau kullanımı: validation loss'u izler ve iyileşme olmazsa learning rate'i azaltır



# Early Stopping sınıfı
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        """
        Args:
            patience (int): İyileşme olmadan beklenebilecek epoch sayısı.
            verbose (bool): Early stopping ile ilgili mesajları yazdır.
        """
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



%%time
# Optimizasyon ve kayıp fonksiyonu
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vit_model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Modeli doğru cihaza taşı
vit_model = vit_model.to(device)

# Early Stopping oluştur
early_stopping = EarlyStopping(patience=40, verbose=True)

# Modeli eğit
history = []

num_epochs = 10
for epoch in range(num_epochs):
    vit_model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vit_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_predictions += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    """epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")"""

    # Eğitim istatistikleri
    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples

    # Validation performansı
    val_loss, val_accuracy = evaluate_model(vit_model, val_loader, criterion)

    # Learning rate scheduler
    scheduler.step(val_loss)

    # Early stopping kontrolü
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    # Epoch sonuçları
    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })

    print(f"Epoch [{epoch+1}/10] | "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

print("Model başarıyla eğitildi!")


# Modeli test et

test_loss, test_accuracy = test_model(vit_model, test_loader, criterion)


# Modeli kaydet
save_model(vit_model, '/kaggle/working/vit_deepfake.pth')


# Performans Değerlendirme
vit_model.eval()
y_true = []
y_pred = []
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = vit_model(images)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_true, y_pred):.2f}")


from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix ve Classification Report 2
y_true = []
y_pred = []

vit_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit_model(images)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
cr = classification_report(y_true, y_pred)
print("Classification Report:\n", cr)


import torch
import cv2
import os
from torchvision import transforms
from PIL import Image
import numpy as np

def test_video(video_path, vit_model, transform, device, frame_skip=10):
    """
    Bir video için deepfake testi yapar. (ViT Modeli İçin)
    
    Args:
        video_path (str): Test edilecek videonun yolu.
        model (torch.nn.Module): Eğitimli ViT modeli.
        transform (torchvision.transforms.Compose): Görüntü dönüşümleri.
        device (torch.device): CUDA veya CPU.
        frame_skip (int): Kaç karede bir işlem yapılacağı.
        
    Returns:
        str: Video tahmini ('REAL' veya 'FAKE').
        float: Güven puanı (0 ile 1 arasında).
    """
    vit_model.eval()  # Modeli değerlendirme moduna al
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Video sonu
        
        frame_count += 1
        
        # Kare atlama kontrolü
        if frame_count % frame_skip != 0:
            continue

        # Gri tonlamaya çevir ve yüz tespiti
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

        # Yüz tespit edilirse işlem yap
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            
            # Görüntüyü PIL'e çevir ve modele uygun formata dönüştür
            face_pil = Image.fromarray(face_resized).convert('RGB')
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            # ViT modeliyle tahmin yap
            with torch.no_grad():
                outputs = vit_model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predictions.append(predicted_class)
    
    cap.release()

    # Yeterli yüz bulunmazsa sonuç belirsiz
    if len(predictions) == 0:
        return "Unknown", 0.0

    # Sonuçları analiz et
    real_count = predictions.count(0)
    fake_count = predictions.count(1)
    confidence = fake_count / len(predictions)
    
    if fake_count > real_count:
        return "FAKE", confidence
    else:
        return "REAL", 1 - confidence


# Test videosu
test_video_path = '/kaggle/input/dfdc-48/dfdc_train_part_48/eukxkhumll.mp4'

# Test işlemi
result, confidence = test_video(test_video_path, vit_model, transform, device)

print(f"Tahmin: {result}, Güven Puanı: {confidence:.2f}")



import matplotlib.pyplot as plt

# Performans verilerinden loss ve accuracy değerlerini ayıklayın
epochs = [h['epoch'] for h in history]
train_loss = [h['train_loss'] for h in history]
val_loss = [h['val_loss'] for h in history]
train_accuracy = [h['train_accuracy'] for h in history]
val_accuracy = [h['val_accuracy'] for h in history]

# Kayıp grafiği (Loss)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Eğitim Kaybı", marker='o')
plt.plot(epochs, val_loss, label="Validasyon Kaybı", marker='o')
plt.title("Eğitim ve Validasyon Kaybı")
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.legend()
plt.grid(True)
plt.show()

# Doğruluk grafiği (Accuracy)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label="Eğitim Doğruluğu", marker='o')
plt.plot(epochs, val_accuracy, label="Validasyon Doğruluğu", marker='o')
plt.title("Eğitim ve Validasyon Doğruluğu")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.legend()
plt.grid(True)
plt.show()


from sklearn.metrics import roc_auc_score, f1_score
import torch

def calculate_auc_f1(model, data_loader, device):
    """
    Modelin AUC ve F1 skorunu hesaplar.
    
    Args:
        model: Eğitilmiş model.
        data_loader: Değerlendirilecek veri yükleyici (train_loader veya val_loader).
        device: Kullanılan cihaz (CPU veya GPU).
    
    Returns:
        auc_score: AUC değeri.
        f1: F1 skoru.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 3:
                images, labels, _ = batch  # Eğer fazladan veri varsa onu görmezden gel.
            else:
                raise ValueError("Unexpected number of elements in batch.")
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return auc, f1




# Eğitim seti üzerinde hesaplama
train_auc, train_f1 = calculate_auc_f1(vit_model, train_loader, device)
print(f"Training AUC: {train_auc:.4f}, Training F1 Score: {train_f1:.4f}")



# Validation seti üzerinde hesaplama
val_auc, val_f1 = calculate_auc_f1(vit_model, val_loader, device)
print(f"Validation AUC: {val_auc:.4f}, Validation F1 Score: {val_f1:.4f}")


# Test seti üzerinde hesaplama
test_auc, test_f1 = calculate_auc_f1(vit_model, test_loader, device)
print(f"Test AUC: {test_auc:.4f}, Test F1 Score: {test_f1:.4f}")




