import os
from PIL import Image
import pydicom

# import other libraries


def convert_dicom_to_jpg(dicom_path, jpg_path):
    """Converts a single DICOM file to a JPEG file with error handling."""
    try:
        # Read the DICOM file
        ds = pydicom.dcmread(dicom_path)
        
        # Extract pixel data
        image = ds.pixel_array.astype(float)
        
        # Apply rescale slope and intercept if available (commonly for CT images)
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            image = image * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        
        # Normalize the pixel values to the range 0-255
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val != 0:
            image_normalized = (image - min_val) / (max_val - min_val) * 255.0
        else:
            image_normalized = np.zeros_like(image)
        
        image_normalized = image_normalized.astype(np.uint8)
        
        # Create a PIL image and save as JPEG
        im = Image.fromarray(image_normalized)
        im.save(jpg_path)
        print(f"Converted {dicom_path} to {jpg_path}")
        
    except RuntimeError as e:
        # This error likely indicates that decompression failed because of missing plugins.
        print(f"RuntimeError for file {dicom_path}: {e}. Skipping this file.")
    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An error occurred while converting {dicom_path}: {e}. Skipping this file.")


def process_folder(input_folder, output_folder):
    """
    Recursively processes the input_folder, converts all .dcm files,
    and saves them into the output_folder, preserving the subfolder structure.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                # Construct the full input file path
                dicom_path = os.path.join(root, file)
                
                # Determine the relative path to recreate folder structure in output_folder
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output file path by replacing .dcm with .jpg
                jpg_filename = os.path.splitext(file)[0] + '.jpg'
                jpg_path = os.path.join(output_dir, jpg_filename)
                
                # Convert and save the image, with error handling inside the conversion function
                convert_dicom_to_jpg(dicom_path, jpg_path)


# Define the input and output directories
input_folder = '/kaggle/input/rsna-breast-cancer-detection/train_images'         # folder containing subfolders with DICOM files
output_folder = '/kaggle/working/jpeg_images'        # folder to store converted JPEG images

# Process the folder
process_folder(input_folder, output_folder)


import pandas as pd
import os

# 1. اقرأ ملف train.csv
df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')

# 2. اضف عمود جديد لمسار الصورة
# لاحظ إن كل صورة موجودة في فولدر اسمه برقم study_id
df['image_path'] = df.apply(lambda row: 
                            os.path.join('/kaggle/input/rsna-breast-cancer-detection/train_images',
                                         str(row['patient_id']), 
                                         str(row['image_id']) + '.dcm'), axis=1)

# 3. عرض أول 5 صفوف للتأكد
print(df.head())



from sklearn.model_selection import train_test_split

# أولاً: قسم Train (80%) والباقي (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cancer'])

# ثانيًا: قسم الباقي (20%) إلى Validation (10%) وTest (10%)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cancer'])

# عرض أحجام الداتا عشان تتأكد
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. تعريف التحويلات المطلوبة
transform = transforms.Compose([
    transforms.Resize((512, 512)),         # تغيير حجم الصورة
    transforms.ToTensor(),                 # تحويل الصورة إلى Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # تطبيع الصورة
])

# 2. كلاس Dataset لتحميل الصور من DataFrame
class RSNADataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # احصل على مسار الصورة والـ label من الـ DataFrame
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['cancer']

        # افتح الصورة باستخدام PIL
        img = Image.open(img_path).convert('RGB')

        # تطبيق التحويلات إذا كانت موجودة
        if self.transform:
            img = self.transform(img)

        return img, label

# 3. تحميل الـ Dataset باستخدام DataLoader مع batch_size المناسب
train_dataset = RSNADataset(train_df, transform=transform)
valid_dataset = RSNADataset(valid_df, transform=transform)
test_dataset = RSNADataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. عرض حجم الدفعة الأولى للتأكد
train_images, train_labels = next(iter(train_loader))
print(f"Train batch size: {train_images.size()}")  # الحجم المتوقع: (32, 3, 512, 512)



import matplotlib.pyplot as plt
import numpy as np

# Function to show a batch of images
def show_sample_batch(loader, num_images=8):
    # Load a batch of images and labels from the DataLoader
    images, labels = next(iter(loader))
    
    # Set up the figure with the required number of subplots (num_images)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # Adjust 2x4 grid for 8 images
    axes = axes.ravel()
    
    # Show the images
    for i in np.arange(num_images):
        img = images[i].numpy().transpose((1, 2, 0))  # Convert from Tensor to HxWxC
        img = np.clip(img, 0, 1)  # Ensure pixel values are between 0 and 1 for visualization
        
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis('off')  # Remove axis for a cleaner look
    
    plt.show()

# Call the function with the train_loader
show_sample_batch(train_loader, num_images=8)  # Display 8 images



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from efficientnet_pytorch import EfficientNet

# 1. تحميل EfficientNetB5
model = EfficientNet.from_pretrained('efficientnet-b5')

# 2. تخصيص الطبقات النهائية لتناسب مهمتنا (2 فئات: سرطان / لا سرطان)
model._fc = nn.Sequential(
    nn.Dropout(p=0.3),  # إضافة طبقة Dropout لتقليل الإفراط في التعميم
    nn.Linear(in_features=model._fc.in_features, out_features=1),  # 1 output node (0 or 1)
    nn.Sigmoid()  # لتوليد القيمة بين 0 و 1 (احتمال السرطان)
)

# 3. اختيار دالة الخسارة (Binary Cross Entropy) + دالة التقييم (Accuracy)
criterion = nn.BCELoss()  # خسارة لثنائية التصنيف
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. تدريب النموذج باستخدام DataLoader و GPU إذا كان متاح
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. دالة تدريب النموذج
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()  # تأكد أن الموديل في وضع التدريب
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # مسح التدرجات السابقة
            
            # تحويل الـ inputs عبر النموذج
            outputs = model(inputs)
            
            loss = criterion(outputs.squeeze(), labels.float())  # حساب الخسارة
            loss.backward()  # حساب التدرجات
            
            optimizer.step()  # تحديث الأوزان
            
            # حساب الدقة
            preds = (outputs.squeeze() > 0.5).float()  # التنبؤات (سرطان أو لا)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# 6. دالة التقييم
def evaluate_model(model, test_loader):
    model.eval()  # وضع التقييم
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():  # تعطيل حساب التدرجات للتقييم
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            preds = (outputs.squeeze() > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    print(f"Test Accuracy: {accuracy:.4f}")

# 7. تدريب النموذج
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# 8. تقييم النموذج
evaluate_model(model, test_loader)



from sklearn.metrics import precision_score, recall_score, f1_score

# دالة لتدريب النموذج مع مراقبة الدقة و الحساسية
def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5):
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()  # وضع النموذج في وضع التدريب
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # مسح التدرجات السابقة
            
            # المرور بالصور عبر النموذج
            outputs = model(inputs)
            
            # حساب الخسارة
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_preds = (np.array(all_preds) > 0.5).astype(int)
        epoch_acc = np.mean(epoch_preds == np.array(all_labels))
        epoch_precision = precision_score(all_labels, epoch_preds)
        epoch_recall = recall_score(all_labels, epoch_preds)
        epoch_f1 = f1_score(all_labels, epoch_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1-Score: {epoch_f1:.4f}")

        # التقييم بعد كل epoch على البيانات التحقق
        model.eval()  # وضع التقييم
        valid_preds = []
        valid_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_preds.extend(outputs.squeeze().cpu().detach().numpy())
                valid_labels.extend(labels.cpu().detach().numpy())
        
        valid_preds = (np.array(valid_preds) > 0.5).astype(int)
        valid_acc = np.mean(valid_preds == np.array(valid_labels))
        
        print(f"Validation Accuracy: {valid_acc:.4f}")

        # حفظ أفضل نموذج بناءً على الدقة
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), 'best_model.pth')

# التدريب مع مراقبة الأداء
train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5)


