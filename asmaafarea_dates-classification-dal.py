# 1. تثبيت المكتبات المطلوبة
!pip install torch torchvision opencv-python matplotlib pandas numpy tqdm scikit-learn >/dev/null 2>&1


# 2. استيراد المكتبات
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report
from torch import nn



# 3. إعداد الإعدادات العامة
class Config:
    SEED = 42
    N_SPLITS = 5  # عدد الطيات = 5
    EPOCHS = 10
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 2
    IMG_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)


# 4. تحميل البيانات
def load_data(train_csv_path, test_images_dir):
    """
    تحميل بيانات التدريب والاختبار.
    
    Args:
        train_csv_path (str): مسار ملف CSV لبيانات التدريب.
        test_images_dir (str): مسار مجلد صور الاختبار.
    
    Returns:
        tuple: يحتوي على DataFrames لبيانات التدريب والاختبار.
    """
    # تحميل بيانات التدريب
    train_data = pd.read_csv(train_csv_path)
    train_data["image_path"] = train_data["filename"].apply(
        lambda x: os.path.join("/kaggle/input/open-data-day-2025-dates-types-classification/train", x)
    )
    train_data = train_data[["image_path", "label"]]

    # تحميل بيانات الاختبار
    test_image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_image_paths = [os.path.join(test_images_dir, f) for f in test_image_files]
    test_data = pd.DataFrame({"image_path": test_image_paths})

    return train_data, test_data


# 5. تحليل البيانات الاستكشافي
def explore_data(train_data):
    """
    تحليل البيانات الاستكشافي.
    
    Args:
        train_data (pd.DataFrame): بيانات التدريب.
    """
    # عرض أول 5 صفوف من بيانات التدريب
    print("أول 5 صفوف من بيانات التدريب:")
    print(train_data.head())

    # عرض توزيع الفئات
    print("\nتوزيع الفئات في بيانات التدريب:")
    print(train_data["label"].value_counts())

    # رسم توزيع الفئات
    plt.figure(figsize=(10, 6))
    train_data["label"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("توزيع الفئات في بيانات التدريب")
    plt.xlabel("الفئة")
    plt.ylabel("عدد الصور")
    plt.xticks(rotation=45)
    plt.show()

    # عرض عينة من الصور مع التسميات
    print("\nعرض عينة من الصور مع التسميات:")
    display_sample_images(train_data, num_samples=5)

def display_sample_images(train_data, num_samples=5):
    """
    عرض عينة من الصور مع التسميات.
    
    Args:
        train_data (pd.DataFrame): بيانات التدريب.
        num_samples (int): عدد الصور المراد عرضها.
    """
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # اختيار صورة عشوائية
        idx = random.randint(0, len(train_data) - 1)
        image_path = train_data.iloc[idx]["image_path"]
        label = train_data.iloc[idx]["label"]

        # قراءة الصورة
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # تحويل إلى RGB

        # عرض الصورة مع التسمية
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()


# 6. تحويلات الصور
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=Config.IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 7. فئة Dataset
class DatesDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_path = row["image_path"]
        label_idx = row["label_idx"] if "label_idx" in row else None

        # قراءة الصورة وتحويلها إلى RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # تطبيق التحويلات
        if self.transforms:
            image = self.transforms(image)

        if self.mode != "test":
            return image, label_idx
        else:
            return image


# 8. بناء النموذج
def get_model(num_classes=7, pretrained=True):
    model = torchvision.models.efficientnet_b4(pretrained=pretrained)
    model.classifier[1] = nn.Linear(1792, num_classes)
    return model


# 9. وظائف التدريب والتحقق
def train_one_epoch(model, optimizer, dataloader, device, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss

def valid_one_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_mapping.keys()))
    return epoch_loss, accuracy


# 10. تدريب النموذج
def run_training(fold, train_data, label_mapping):
    print(f"////////// Fold: {fold} ///////////")

    # تقسيم البيانات
    train_data["fold"] = -1
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
    for fold_number, (tr_idx, val_idx) in enumerate(skf.split(train_data, train_data["label_idx"])):
        train_data.loc[val_idx, "fold"] = fold_number

    train_data_fold = train_data[train_data["fold"] != fold].reset_index(drop=True)
    valid_data_fold = train_data[train_data["fold"] == fold].reset_index(drop=True)

    # تحميل البيانات
    train_dataset = DatesDataset(train_data_fold, mode="train", transforms=train_transforms)
    valid_dataset = DatesDataset(valid_data_fold, mode="valid", transforms=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=Config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.VALID_BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)

    # بناء النموذج
    model = get_model(num_classes=len(label_mapping), pretrained=True)
    model.to(Config.DEVICE)

    # تحديد الخسارة والمحسن
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        print(f"\nFold {fold} | Epoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = train_one_epoch(model, optimizer, train_loader, Config.DEVICE, criterion)
        valid_loss, valid_acc = valid_one_epoch(model, valid_loader, Config.DEVICE, criterion)
        print(f"  [Train Loss: {train_loss:.4f}]  [Valid Loss: {valid_loss:.4f}]  [Valid Acc: {valid_acc:.4f}]")

        # حفظ أفضل نموذج
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = f"effb4_fold_{fold}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  --> Model saved to {save_path}")

        # ضبط معدل التعلم
        scheduler.step(valid_acc)

    print(f"\nFold {fold} best accuracy: {best_acc:.4f}")


# 11. التنبؤ بالبيانات
def run_inference(test_data, label_mapping):
    test_dataset = DatesDataset(test_data, mode="test", transforms=valid_transforms)
    test_loader = DataLoader(test_dataset, batch_size=Config.VALID_BATCH_SIZE,
                             shuffle=False, num_workers=Config.NUM_WORKERS)

    fold_preds = []
    for fold in range(Config.N_SPLITS):
        model_path = f"effb4_fold_{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: no model found at {model_path}, skipping this fold.")
            continue

        model = get_model(num_classes=len(label_mapping), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)

        model.eval()
        preds = []
        with torch.no_grad():
            for imgs in tqdm(test_loader, desc="Inferring", leave=False):
                imgs = imgs.to(Config.DEVICE)
                outputs = model(imgs)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                preds.append(probabilities)
        fold_preds.append(np.concatenate(preds, axis=0))

    # متوسط التنبؤات
    final_preds = np.mean(fold_preds, axis=0)
    class_indices = np.argmax(final_preds, axis=1)

    inv_map = {v: k for k, v in label_mapping.items()}
    final_labels = [inv_map[i] for i in class_indices]

    return final_labels


# 12. حفظ النتائج

def save_submission(test_data, predictions, output_folder):
    submission_df = pd.DataFrame({
        "filename": test_data["image_path"].apply(os.path.basename),
        "label": predictions
    })
    os.makedirs(output_folder, exist_ok=True)
    submission_df.to_csv(os.path.join(output_folder, "submission.csv"), index=False)
    print("Submission file saved successfully!")


# 13. عرض عينة من التنبؤات مع الصور

def display_predictions_with_images(test_data, predictions, num_samples=5):
    """
    عرض عينة من الصور مع التسميات المتوقعة.
    
    Args:
        test_data (pd.DataFrame): بيانات الاختبار.
        predictions (list): قائمة بالتسميات المتوقعة.
        num_samples (int): عدد الصور المراد عرضها.
    """
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        # اختيار صورة عشوائية
        idx = random.randint(0, len(test_data) - 1)
        image_path = test_data.iloc[idx]["image_path"]
        predicted_label = predictions[idx]

        # قراءة الصورة
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # تحويل إلى RGB

        # عرض الصورة مع التسمية المتوقعة
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis("off")
    plt.show()


# 14. التنفيذ الرئيسي

# المسارات
train_csv_path = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
test_images_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/test"
output_folder = "/kaggle/working/Submissions"

# تحميل البيانات
train_data, test_data = load_data(train_csv_path, test_images_dir)

# تحويل التسميات النصية إلى أرقام
label_mapping = {
    "Ajwa": 0,
    "Medjool": 1,
    "Meneifi": 2,
    "Nabtat Ali": 3,
    "Shaishe": 4,
    "Sokari": 5,
    "Sugaey": 6
}
train_data["label_idx"] = train_data["label"].map(label_mapping)


# تحليل البيانات الاستكشافي
explore_data(train_data)


# تدريب النموذج
for fold in range(Config.N_SPLITS):
    run_training(fold, train_data, label_mapping)


# التنبؤ بالبيانات
final_predictions = run_inference(test_data, label_mapping)


# حفظ النتائج
save_submission(test_data, final_predictions, output_folder)


# عرض عينة من التنبؤات مع الصور
print("\nعرض عينة من التنبؤات مع الصور:")
display_predictions_with_images(test_data, final_predictions, num_samples=5)

