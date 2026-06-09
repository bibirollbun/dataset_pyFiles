import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- 1. Định nghĩa đường dẫn ---
# Đường dẫn đến thư mục chứa ảnh training
IMAGE_DIR = '/kaggle/input/modia-ml-2024/train/train/'
# Đường dẫn đến file CSV chứa nhãn
CSV_PATH = '/kaggle/input/modia-ml-2024/train.csv'

# --- 2. Định nghĩa ánh xạ nhãn ---
# Ánh xạ từ số label sang tên nhãn
LABEL_MAP = {
    1: 'cauliflower',
    2: 'mountain tent',
    3: 'head cabbage',
    4: 'honeycomb'
}

# --- 3. Đọc và xử lý file CSV ---
try:
    # Đọc file CSV. Lưu ý cột đầu tiên (index_col=0) là chỉ số (STT) của ảnh
    # Trong trường hợp file CSV không có header, ta sẽ đặt tên cột thủ công
    df = pd.read_csv(CSV_PATH)

    # Đổi tên cột để dễ làm việc hơn (giả định cột 1 là index, cột 2 là label_id)
    # Dựa trên mô tả: "cột 1 là stt của ảnh cột 2 là label được đánh số tương ứng"
    # Nếu file CSV có 2 cột và không có header, code sẽ là:
    # df.columns = ['image_index', 'label_id']
    
    # Giả sử file CSV có header mặc định:
    df.rename(columns={df.columns[0]: 'image_index', df.columns[1]: 'label_id'}, inplace=True)
    
    # Thêm cột tên nhãn dựa trên ánh xạ
    df['label_name'] = df['label_id'].map(LABEL_MAP)
    
    # Thêm cột tên file ảnh (dựa trên quy ước tên file là "STT.jpg")
    df['filename'] = df['image_index'].astype(str) + '.jpg'
    
    # Thêm cột đường dẫn đầy đủ đến file ảnh
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file CSV tại {CSV_PATH}")
    exit()

# --- 4. Hiển thị thông tin DataFrame kết quả ---
print("--- DataFrame chứa thông tin nhãn và đường dẫn ảnh ---")
print(df.head())
print(f"\nTổng số ảnh được đọc: {len(df)}")


# --- 5. Ví dụ: Đọc và hiển thị một ảnh cụ thể ---
# Lấy đường dẫn của ảnh đầu tiên trong DataFrame
first_image_path = df.iloc[0]['filepath']
first_image_label = df.iloc[0]['label_name']

print(f"\nVí dụ đọc file: {first_image_path}")
print(f"Nhãn: {first_image_label}")

try:
    # Đọc ảnh
    img = mpimg.imread(first_image_path)
    
    # Hiển thị ảnh (chỉ chạy được trong môi trường Jupyter/Kaggle Notebook)
    plt.imshow(img)
    plt.title(f"Label: {first_image_label}")
    plt.axis('off')
    plt.show()
    
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file ảnh tại {first_image_path}")
except Exception as e:
    print(f"Lỗi khi đọc ảnh: {e}")

# --- Kết quả cuối cùng là DataFrame 'df' ---
# 'df' chứa tất cả thông tin bạn cần để xây dựng


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Thêm đoạn code này ngay sau khi bạn load DataFrame `df`
print("Giá trị nhãn ban đầu:", df['label_id'].unique())

# Ánh xạ lại nhãn bằng cách trừ đi 1
df['label_id'] = df['label_id'] - 1

print("Giá trị nhãn mới sau khi chuẩn hóa:", df['label_id'].unique())
print("Giá trị nhỏ nhất mới:", df['label_id'].min())
print("Giá trị lớn nhất mới:", df['label_id'].max())


import pandas as pd
from sklearn.model_selection import train_test_split

# Giả sử df là DataFrame của bạn đã được load
# df = pd.read_csv(...) 

# Chia dữ liệu thành 80% train và 20% validation
# Sử dụng `stratify=df['label_id']` để đảm bảo tỷ lệ các lớp trong hai tập là như nhau
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label_id']
)

print(f"Số lượng mẫu train: {len(train_df)}")
print(f"Số lượng mẫu validation: {len(val_df)}")


import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm # Dùng tqdm.notebook cho Kaggle

# --- Cấu hình chung ---
TEST_DIR = '/kaggle/input/modia-ml-2024/test/test/'
IMG_SIZE = 224 # Phải giống hệt lúc training
BATCH_SIZE = 64 # Có thể tăng batch size khi test vì không cần tính gradient

# Lấy danh sách tất cả các file ảnh trong thư mục test
test_filenames = os.listdir(TEST_DIR)

# Tạo DataFrame cho dữ liệu test
test_df = pd.DataFrame({'filename': test_filenames})

# Trích xuất cột 'stt' từ filename và chuyển thành số nguyên để sắp xếp cho đúng thứ tự
test_df['id'] = test_df['filename'].apply(lambda x: int(x.split('.')[0]))

# Sắp xếp DataFrame theo 'stt' để đảm bảo thứ tự dự đoán là chính xác
test_df = test_df.sort_values(by='id').reset_index(drop=True)

# Tạo đường dẫn đầy đủ
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(TEST_DIR, x))

print(f"Tìm thấy {len(test_df)} ảnh trong tập test.")
test_df.head()


# Map từ output của model (0,1,2,3) về label gốc (1,2,3,4) để submit
# Dựa trên việc bạn đã làm `label = label - 1`
# Nếu bạn dùng LabelEncoder, bạn cần dùng `le.inverse_transform`
REVERSE_LABEL_MAP = {
    0: 1,
    1: 2,
    2: 3,
    3: 4
}


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- Các tham số cấu hình ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = df['label_id'].nunique()

# --- 1. Định nghĩa các phép biến đổi (Transforms) ---

# Chuẩn hóa theo giá trị mean và std của ImageNet
# Đây là bước BẮT BUỘC cho các model pre-trained
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Transforms cho tập train (bao gồm augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(), # Chuyển ảnh PIL thành Tensor và scale về [0, 1]
    normalize,           # Chuẩn hóa
])

# Transforms cho tập validation (không có augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])


# --- 2. Tạo lớp Dataset tùy chỉnh ---
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        # Trả về tổng số lượng mẫu
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Lấy đường dẫn ảnh và nhãn tại vị trí idx
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label_id']
        
        # Mở ảnh bằng PIL
        image = Image.open(img_path).convert("RGB")
        
        # Áp dụng các phép biến đổi nếu có
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# --- 3. Tạo các đối tượng Dataset ---
train_dataset = CustomImageDataset(train_df, transform=train_transforms)
val_dataset = CustomImageDataset(val_df, transform=val_transforms)

# --- 4. Tạo các đối tượng DataLoader ---
# DataLoader sẽ giúp chúng ta tạo batch, xáo trộn dữ liệu và tải song song
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,      # Xáo trộn dữ liệu ở mỗi epoch
    num_workers=2      # Sử dụng 2 tiến trình để tải dữ liệu, tăng tốc độ
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,     # Không cần xáo trộn tập validation
    num_workers=2
)

# --- Kiểm tra một batch ---
print("\n--- PyTorch DataLoader ---")
images, labels = next(iter(train_loader))
print("Kích thước batch ảnh:", images.shape)
print("Kích thước batch nhãn:", labels.shape)
# images.shape sẽ là (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
# labels.shape sẽ là (BATCH_SIZE,)

# Sẵn sàng cho vòng lặp training
# for images, labels in train_loader:
#     # ... training code ...


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm # Để có thanh tiến trình đẹp mắt

# --- Các tham số ---
NUM_CLASSES = 4
LEARNING_RATE_FT = 1e-3
LEARNING_RATE_FN = 1e-5
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Tải model pre-trained ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# --- 2. Đóng băng các tham số của base model ---
for param in model.parameters():
    param.requires_grad = False

# --- 3. Thay thế classifier head ---
# Lấy số lượng features đầu vào của lớp classifier cũ
num_ftrs = model.classifier[1].in_features

# Tạo classifier mới
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

model = model.to(DEVICE)

# --- GIAI ĐOẠN 1: FEATURE EXTRACTION ---
print("\n--- BẮT ĐẦU GIAI ĐOẠN 1: FEATURE EXTRACTION ---")

# Chỉ các tham số của classifier mới cần được tối ưu
params_to_update = model.classifier.parameters()
optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE_FT)
criterion = nn.CrossEntropyLoss() # Tương đương sparse_categorical_crossentropy

# Hàm training loop (có thể tách ra để tái sử dụng)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

        epoch_acc = corrects.double() / len(val_loader.dataset)
        print(f"Epoch {epoch+1} Acc: {epoch_acc:.4f}")

        # Lưu model tốt nhất
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model_feature_extraction.pth')

# Bắt đầu training giai đoạn 1
# train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)


# --- GIAI ĐOẠN 2: FINE-TUNING ---
print("\n--- BẮT ĐẦU GIAI ĐOẠN 2: FINE-TUNING ---")

# Tải lại trọng số tốt nhất từ giai đoạn 1
model.load_state_dict(torch.load('best_model_feature_extraction.pth'))

# Mở băng tất cả các lớp
for param in model.parameters():
    param.requires_grad = True

# Tạo optimizer mới với learning rate RẤT THẤP cho toàn bộ model
optimizer_fine_tuning = optim.Adam(model.parameters(), lr=LEARNING_RATE_FN)

# Tiếp tục training giai đoạn 2
# train_model(model, train_loader, val_loader, criterion, optimizer_fine_tuning, num_epochs=EPOCHS)
torch.save(model.state_dict(), 'best_model_fine_tuned.pth')


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# --- 1. Tạo Dataset và DataLoader cho tập test ---

# Transforms cho tập test phải giống hệt transforms cho tập validation
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset tùy chỉnh cho tập test (chỉ trả về ảnh)
class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = TestImageDataset(test_df, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 2. Tải model đã được huấn luyện ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4 # Phải khớp lúc train

# Xây dựng lại kiến trúc model
model = models.efficientnet_b0(weights=None) # Không cần tải pre-trained weights nữa
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3, inplace=True),
    torch.nn.Linear(num_ftrs, NUM_CLASSES)
)

# Tải trọng số đã lưu
model_path = '/kaggle/working/best_model_exp1.pth'
print(f"Đang tải model từ: {model_path}")
model.load_state_dict(torch.load(model_path))
model = model.to(DEVICE)

# --- 3. Thực hiện dự đoán ---
print("Bắt đầu dự đoán trên tập test...")
model.eval() # Bật chế độ evaluation, rất quan trọng!
all_preds = []

with torch.no_grad(): # Không cần tính gradient khi inference
    for images in tqdm(test_loader, desc="Dự đoán"):
        images = images.to(DEVICE)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Chuyển predictions về CPU và thêm vào list
        all_preds.extend(predicted.cpu().numpy())

# --- 4. Tạo file submission ---
# Thêm kết quả dự đoán (0,1,2,3) vào dataframe
test_df['predicted_index'] = all_preds

# Ánh xạ ngược lại thành nhãn 1,2,3,4
test_df['label'] = test_df['predicted_index'].apply(lambda x: REVERSE_LABEL_MAP[x])
# Hoặc cách đơn giản hơn: test_df['class'] = test_df['predicted_index'] + 1

# Tạo dataframe submission cuối cùng
submission_df = test_df[['id', 'label']]

# Lưu ra file CSV
submission_df.to_csv('submission.csv', index=False)

print("\nĐã tạo file submission.csv thành công!")
submission_df.head()


import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm

# Giả sử df đã được load và có các cột 'filepath', 'label_name'
# df = ...

# Thiết lập style cho biểu đồ
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


print("Số lượng ảnh mỗi lớp:")
print(df['label_name'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(y=df['label_name'], order=df['label_name'].value_counts().index)
plt.title('Phân bố số lượng ảnh của các lớp')
plt.xlabel('Số lượng ảnh')
plt.ylabel('Lớp')
plt.show()


# Hàm để lấy kích thước ảnh
def get_image_dimensions(filepath):
    try:
        with Image.open(filepath) as img:
            return img.size # (width, height)
    except Exception as e:
        print(f"Lỗi khi đọc file {filepath}: {e}")
        return (None, None)

# Áp dụng hàm này cho toàn bộ dataframe (có thể mất vài giây)
# tqdm giúp hiển thị thanh tiến trình
tqdm.pandas(desc="Đọc kích thước ảnh")
df[['width', 'height']] = df['filepath'].progress_apply(lambda x: pd.Series(get_image_dimensions(x)))

# Bỏ qua các ảnh bị lỗi nếu có
df_dims = df.dropna(subset=['width', 'height']).copy()
df_dims['aspect_ratio'] = df_dims['width'] / df_dims['height']

# --- Vẽ biểu đồ ---

# 1. Biểu đồ phân tán của Width vs Height
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df_dims, x='width', y='height', hue='label_name', alpha=0.5)
plt.title('Phân bố Kích thước ảnh (Width vs Height)')
plt.xlabel('Chiều rộng (pixels)')
plt.ylabel('Chiều cao (pixels)')
plt.show()

# 2. Biểu đồ phân bố của Tỷ lệ khung hình (Aspect Ratio)
plt.figure(figsize=(10, 5))
sns.histplot(data=df_dims, x='aspect_ratio', bins=50, kde=True)
plt.axvline(1.0, color='red', linestyle='--', label='Ảnh vuông (ratio=1.0)')
plt.title('Phân bố Tỷ lệ khung hình (Width / Height)')
plt.xlabel('Tỷ lệ khung hình')
plt.legend()
plt.show()


def show_sample_images_by_class(df, class_name, n_samples=5):
    sample_df = df[df['label_name'] == class_name].sample(n_samples)
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Các ảnh mẫu của lớp: {class_name.upper()}', fontsize=16)
    
    for i, row in enumerate(sample_df.itertuples()):
        plt.subplot(1, n_samples, i + 1)
        img = Image.open(row.filepath)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Size: {img.size}")
    plt.show()

for label_name in df['label_name'].unique():
    show_sample_images_by_class(df, label_name)


# ==========================================================
# THỬ NGHIỆM 1: TIẾP TỤC TRAINING VỚI RANDOMRESIZEDCROP
# ==========================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# --- 1. CẤU HÌNH ---
# Giả sử các dataframe train_df và val_df đã tồn tại từ các cell trước

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = df['label_id'].nunique() # Đảm bảo df đã được load
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Đường dẫn đến model cũ và tên cho model mới
PREV_MODEL_PATH = 'best_model_fine_tuned.pth' # Tải từ model tốt nhất lần trước
NEW_MODEL_PATH = 'best_model_exp1.pth'      # Tên file model mới cho thử nghiệm này

# Cấu hình cho việc training tiếp
CONTINUE_TRAIN_EPOCHS = 20
LEARNING_RATE = 1e-5 # Bắt đầu với learning rate thấp của giai đoạn fine-tuning

# --- 2. ĐỊNH NGHĨA LẠI TRANSFORMS ---
print("--- Bước 1: Định nghĩa lại chiến lược Augmentation ---")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transforms_exp1 = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), # THAY ĐỔI CHÍNH
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    normalize,
])

# Giữ nguyên val_transforms để so sánh công bằng
val_transforms_exp1 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])
print("Sử dụng RandomResizedCrop cho tập train.")

# --- 3. TẠO LẠI DATASET & DATALOADER ---
print("\n--- Bước 2: Tạo lại Dataset và DataLoader ---")

# Giả sử class CustomImageDataset đã được định nghĩa
# Nếu không, hãy định nghĩa lại nó ở đây
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label_id']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset_exp1 = CustomImageDataset(train_df, transform=train_transforms_exp1)
val_dataset_exp1 = CustomImageDataset(val_df, transform=val_transforms_exp1)

train_loader_exp1 = DataLoader(train_dataset_exp1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader_exp1 = DataLoader(val_dataset_exp1, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print("DataLoader đã sẵn sàng.")

# --- 4. KHỞI TẠO MODEL VÀ TẢI TRỌNG SỐ CŨ ---
print(f"\n--- Bước 3: Tải trọng số từ '{PREV_MODEL_PATH}' ---")
model = models.efficientnet_b0(weights=None) # Khởi tạo kiến trúc
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

try:
    model.load_state_dict(torch.load(PREV_MODEL_PATH))
    print("Tải trọng số thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{PREV_MODEL_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    # Dừng cell nếu không có file weight
    raise

model = model.to(DEVICE)

# --- 5. TIẾP TỤC HUẤN LUYỆN ---
print("\n--- Bước 4: Bắt đầu quá trình huấn luyện tiếp (fine-tuning) ---")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=CONTINUE_TRAIN_EPOCHS, eta_min=1e-7)

best_acc = 0.0
# Tải lại acc tốt nhất của lần trước để so sánh
# (Nếu bạn đã ghi lại, nếu không thì bắt đầu từ 0)
# best_acc = 0.9863 

for epoch in range(CONTINUE_TRAIN_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader_exp1, desc=f"Epoch {epoch+1}/{CONTINUE_TRAIN_EPOCHS} [Training]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader_exp1, desc=f"Epoch {epoch+1}/{CONTINUE_TRAIN_EPOCHS} [Validation]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

    epoch_acc = corrects.double() / len(val_dataset_exp1)
    print(f"Epoch {epoch+1} | Val Acc: {epoch_acc:.4f} | Current LR: {scheduler.get_last_lr()[0]:.1e}")

    # Lưu model tốt nhất
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), NEW_MODEL_PATH)
        print(f"*** New best model saved to '{NEW_MODEL_PATH}' with accuracy: {best_acc:.4f} ***")

    # Cập nhật learning rate
    scheduler.step()

print(f"\n--- Hoàn thành Thử nghiệm 1 ---")
print(f"Model tốt nhất đã được lưu tại '{NEW_MODEL_PATH}' với Validation Accuracy cao nhất là: {best_acc:.4f}")


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# ==========================================================
# THỬ NGHIỆM 2: HUẤN LUYỆN EFFICIENTNETB3 VỚI IMG_SIZE 300
# ==========================================================

# --- 1. CẤU HÌNH ---
# Giả sử các dataframe df, train_df, val_df đã tồn tại từ các cell trước
# df = pd.read_csv(...)
# train_df, val_df = train_test_split(...)

# Cấu hình model và ảnh
IMG_SIZE = 300  # Kích thước khuyến nghị cho EfficientNetB3
MODEL_ARCH = 'efficientnet_b3'
MODEL_WEIGHTS = models.EfficientNet_B3_Weights.DEFAULT

# Cấu hình training
# Giảm batch size vì model và ảnh lớn hơn, tốn nhiều VRAM hơn
BATCH_SIZE = 16 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cấu hình các giai đoạn
EPOCHS_FT = 15  # Giai đoạn Feature Extraction
EPOCHS_FN = 25  # Giai đoạn Fine-tuning
LR_FT = 1e-3
LR_FN = 2e-5

# Đường dẫn lưu model
NEW_MODEL_PATH = 'best_model_b3_exp2.pth'
NUM_CLASSES = df['label_id'].nunique()

print(f"--- Bắt đầu Thử nghiệm 2 ---")
print(f"Model: {MODEL_ARCH}, Image Size: {IMG_SIZE}x{IMG_SIZE}, Device: {DEVICE}")

# --- 2. TRANSFORMS & DATALOADER ---
print("\n--- Bước 1: Chuẩn bị Transforms và DataLoaders ---")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),  # Một chiến lược augmentation mạnh và tự động
    transforms.ToTensor(),
    normalize,
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# Giả sử class CustomImageDataset đã được định nghĩa
# Nếu không, hãy định nghĩa lại nó ở đây
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label_id']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
        
train_dataset = CustomImageDataset(train_df, transform=train_transforms)
val_dataset = CustomImageDataset(val_df, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoaders đã sẵn sàng.")

# --- 3. HÀM TRAINING TÁI SỬ DỤNG ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10, model_path='best_model.pth'):
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_acc = corrects.double() / len(val_dataset)
        
        if scheduler:
            lr_to_print = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            lr_to_print = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Val Acc: {epoch_acc:.4f} | LR: {lr_to_print:.1e}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), model_path)
            print(f"*** New best model saved to '{model_path}' with accuracy: {best_acc:.4f} ***")
    
    print(f"Training phase complete. Best Val Acc: {best_acc:.4f}")
    return best_acc

# --- 4. GIAI ĐOẠN 1: FEATURE EXTRACTION ---
print(f"\n--- Bắt đầu Giai đoạn 1: Feature Extraction cho {MODEL_ARCH} ---")
model = models.get_model(MODEL_ARCH, weights=MODEL_WEIGHTS)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True), # Tăng dropout cho model lớn hơn
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(DEVICE)

optimizer_ft = optim.AdamW(model.classifier.parameters(), lr=LR_FT)
criterion = nn.CrossEntropyLoss()

train_loop(
    model, train_loader, val_loader, criterion, optimizer_ft, 
    num_epochs=EPOCHS_FT, model_path=NEW_MODEL_PATH
)

# --- 5. GIAI ĐOẠN 2: FINE-TUNING ---
print(f"\n--- Bắt đầu Giai đoạn 2: Fine-tuning cho {MODEL_ARCH} ---")

# Tải lại model tốt nhất từ Giai đoạn 1 để bắt đầu Fine-tuning
print(f"Tải lại trọng số tốt nhất từ '{NEW_MODEL_PATH}' để bắt đầu fine-tuning.")
model.load_state_dict(torch.load(NEW_MODEL_PATH))

# Mở băng tất cả các lớp
for param in model.parameters():
    param.requires_grad = True

optimizer_fn = optim.AdamW(model.parameters(), lr=LR_FN, weight_decay=1e-3)
scheduler_fn = CosineAnnealingLR(optimizer_fn, T_max=EPOCHS_FN, eta_min=1e-7)

train_loop(
    model, train_loader, val_loader, criterion, optimizer_fn, 
    scheduler=scheduler_fn, num_epochs=EPOCHS_FN, model_path=NEW_MODEL_PATH
)

print(f"\n--- HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ---")
print(f"Model cuối cùng đã được lưu tại '{NEW_MODEL_PATH}'.")


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# ==========================================================
# TẠO FILE SUBMISSION CHO THỬ NGHIỆM 2 (EFFICIENTNETB3)
# ==========================================================

# --- 1. CẤU HÌNH NHẤT QUÁN VỚI LÚC TRAINING ---
print("--- Bước 1: Cấu hình cho việc dự đoán ---")
# THAY ĐỔI QUAN TRỌNG: Phải khớp với cấu hình của Thử nghiệm 2
IMG_SIZE = 300
MODEL_ARCH = 'efficientnet_b3'
MODEL_PATH = '/kaggle/working/swa_model_final.pth' # Đường dẫn đến model B3 đã train

# Cấu hình chung
TEST_DIR = '/kaggle/input/modia-ml-2024/test/test/'
BATCH_SIZE = 32 # Có thể tăng batch size khi test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4 # Số lượng lớp của bài toán

print(f"Model: {MODEL_ARCH}, Image Size: {IMG_SIZE}, Model Path: {MODEL_PATH}")

# --- 2. CHUẨN BỊ DATAFRAME TẬP TEST ---
print("\n--- Bước 2: Chuẩn bị dữ liệu test ---")
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'filename': test_filenames})
test_df['id'] = test_df['filename'].apply(lambda x: int(x.split('.')[0]))
test_df = test_df.sort_values(by='id').reset_index(drop=True)
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(TEST_DIR, x))
print(f"Tìm thấy {len(test_df)} ảnh trong tập test.")

# --- 3. TẠO DATASET VÀ DATALOADER CHO TẬP TEST ---
# Transforms cho tập test phải giống hệt transforms cho tập validation lúc train
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Sử dụng IMG_SIZE mới
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = TestImageDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoader cho tập test đã sẵn sàng.")

# --- 4. TẢI MODEL ĐÃ HUẤN LUYỆN ---
print(f"\n--- Bước 3: Tải model từ '{MODEL_PATH}' ---")

# Xây dựng lại đúng kiến trúc model
model = models.get_model(MODEL_ARCH, weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4, inplace=True),
    torch.nn.Linear(num_ftrs, NUM_CLASSES)
)

# Tải trọng số đã lưu
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Tải trọng số thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{MODEL_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    raise

model = model.to(DEVICE)

# --- 5. THỰC HIỆN DỰ ĐOÁN ---
print("\n--- Bước 4: Bắt đầu dự đoán trên tập test ---")
model.eval() # Bật chế độ evaluation
all_preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Dự đoán"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())

# --- 6. TẠO FILE SUBMISSION ---
print("\n--- Bước 5: Tạo file submission.csv ---")
# Thêm kết quả dự đoán (0,1,2,3) vào dataframe
test_df['predicted_index'] = all_preds

# Ánh xạ ngược lại thành nhãn 1,2,3,4
test_df['label'] = test_df['predicted_index'] + 1

# Tạo dataframe submission cuối cùng
submission_df = test_df[['id', 'label']]

# Lưu ra file CSV
submission_df.to_csv('submission.csv', index=False)

print("\nĐã tạo file submission.csv thành công!")
print("Xem trước 5 dòng đầu tiên:")
print(submission_df.head())


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# ==========================================================
# GIAI ĐOẠN 1: HUẤN LUYỆN THÊM VÀ LƯU CHECKPOINTS CHO SWA
# ==========================================================

# --- 1. CẤU HÌNH ---
# Phải khớp với Thử nghiệm 2
IMG_SIZE = 300
MODEL_ARCH = 'efficientnet_b3'
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4 # df['label_id'].nunique()

# Đường dẫn model để bắt đầu và thư mục để lưu checkpoints
BASE_MODEL_PATH = 'best_model_b3_exp2.pth'
CHECKPOINT_DIR = './swa_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Tạo thư mục nếu chưa có

# Cấu hình training
SWA_EPOCHS = 10
SWA_LR = 5e-6 # Sử dụng một learning rate rất nhỏ, gần như không đổi

print(f"--- Bắt đầu Giai đoạn 1: Chuẩn bị checkpoints cho SWA ---")
print(f"Tải model từ: {BASE_MODEL_PATH}")
print(f"Sẽ train thêm {SWA_EPOCHS} epochs và lưu vào thư mục '{CHECKPOINT_DIR}'")

# --- 2. TRANSFORMS & DATALOADER (Giống hệt lúc train Thử nghiệm 2) ---
# ... (Giả sử train_loader và val_loader cho IMG_SIZE=300 đã được tạo) ...

# --- 3. TẢI MODEL VÀ TIẾP TỤC HUẤN LUYỆN ---
model = models.get_model(MODEL_ARCH, weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model.load_state_dict(torch.load(BASE_MODEL_PATH))
model = model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=SWA_LR)
criterion = nn.CrossEntropyLoss()

# Vòng lặp training để tạo checkpoints
model.train()
for epoch in range(SWA_EPOCHS):
    for inputs, labels in tqdm(train_loader, desc=f"SWA Epoch {epoch+1}/{SWA_EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Lưu checkpoint sau mỗi epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'swa_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Đã lưu checkpoint: {checkpoint_path}")

print("\n--- Hoàn thành Giai đoạn 1: Đã tạo đủ checkpoints. ---")


import os
import torch
import torch.nn as nn
from torchvision import models
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm.notebook import tqdm

# ==========================================================
# GIAI ĐOẠN 2: TỔ HỢP CÁC CHECKPOINTS THÀNH SWA MODEL
# ==========================================================

print(f"\n--- Bắt đầu Giai đoạn 2: Tạo SWA model từ các checkpoints ---")

# --- 1. CẤU HÌNH (Phải khớp với Giai đoạn 1) ---
IMG_SIZE = 300
MODEL_ARCH = 'efficientnet_b3'
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = './swa_checkpoints'
SWA_MODEL_SAVE_PATH = 'swa_model_final.pth'

# --- 2. KHỞI TẠO MODEL GỐC VÀ SWA MODEL ---
# Khởi tạo model với kiến trúc giống hệt lúc train
base_model = models.get_model(MODEL_ARCH, weights=None)
num_ftrs = base_model.classifier[1].in_features
base_model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
base_model = base_model.to(DEVICE)

# Khởi tạo SWA model từ base_model.
# AveragedModel sẽ lưu một bản sao của model và tính trung bình các trọng số được cập nhật vào nó.
swa_model = AveragedModel(base_model)

# --- 3. TẢI VÀ TÍNH TRUNG BÌNH CÁC CHECKPOINTS ---
# Lấy danh sách tất cả các file checkpoint đã lưu
checkpoint_files = sorted([os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])

print(f"Tìm thấy {len(checkpoint_files)} checkpoints để thực hiện SWA.")

# Lặp qua từng checkpoint, tải trọng số và cập nhật vào swa_model
for checkpoint_path in tqdm(checkpoint_files, desc="Averaging checkpoints"):
    # Tải state_dict từ checkpoint
    checkpoint_state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Cập nhật state_dict này vào model cơ sở
    base_model.load_state_dict(checkpoint_state_dict)
    
    # Cập nhật trọng số của swa_model bằng cách lấy trung bình với trọng số mới từ base_model
    swa_model.update_parameters(base_model)

print("Đã hoàn tất việc tính trung bình trọng số.")

# --- 4. CẬP NHẬT BATCH NORMALIZATION STATISTICS ---
# Đây là một bước quan trọng sau khi kết hợp SWA.
# Ta cần cho dữ liệu đi qua model SWA để các lớp BatchNorm tính toán lại giá trị running mean và variance.
# Nếu không có bước này, hiệu năng của model có thể bị giảm sút đáng kể.
print("Bắt đầu cập nhật thống kê Batch Normalization cho SWA model...")

# Chuyển swa_model sang chế độ train để BN được cập nhật
swa_model.train()

# Cần dùng train_loader để cập nhật BN
# Giả sử `train_loader` đã được định nghĩa từ trước
# `update_bn` sẽ tự động chạy một vài batch dữ liệu qua model để cập nhật.
update_bn(train_loader, swa_model, device=DEVICE)

print("Đã cập nhật xong Batch Normalization.")

# --- 5. LƯU SWA MODEL HOÀN CHỈNH ---
# Chuyển model về chế độ eval trước khi lưu
swa_model.eval()

# Lưu lại state_dict của swa_model. Chú ý rằng swa_model là một module bọc ngoài
# nên ta cần lưu `swa_model.module.state_dict()`
torch.save(swa_model.module.state_dict(), SWA_MODEL_SAVE_PATH)

print(f"\n--- Hoàn thành Giai đoạn 2 ---")
print(f"SWA model cuối cùng đã được lưu tại: {SWA_MODEL_SAVE_PATH}")
print("Bây giờ bạn có thể tải model này để đánh giá hoặc dự đoán.")

# --- CÁCH SỬ DỤNG MODEL ĐÃ LƯU ---
#
# # Khởi tạo lại kiến trúc model
# final_model = models.get_model(MODEL_ARCH, weights=None)
# num_ftrs = final_model.classifier[1].in_features
# final_model.classifier = nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(num_ftrs, NUM_CLASSES)
# )
#
# # Tải trọng số đã được SWA
# final_model.load_state_dict(torch.load(SWA_MODEL_SAVE_PATH))
# final_model = final_model.to(DEVICE)
# final_model.eval()
#
# # Bây giờ `final_model` đã sẵn sàng để sử dụng
# # ví dụ: with torch.no_grad():
# #           predictions = final_model(images)


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np

# ==========================================================
# TẠO FILE SUBMISSION VỚI TEST-TIME AUGMENTATION (TTA)
# ==========================================================

# --- 1. CẤU HÌNH NHẤT QUÁN VỚI LÚC TRAINING ---
print("--- Bước 1: Cấu hình cho việc dự đoán ---")
IMG_SIZE = 300
MODEL_ARCH = 'efficientnet_b3'
# Hãy chắc chắn đường dẫn này là chính xác
MODEL_PATH = '/kaggle/working/swa_model_final.pth' 

TEST_DIR = '/kaggle/input/modia-ml-2024/test/test/'
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4

print(f"Model: {MODEL_ARCH}, Image Size: {IMG_SIZE}, Model Path: {MODEL_PATH}")
print(f"Device: {DEVICE}, Sử dụng Test-Time Augmentation: True (Lật ngang)")

# --- 2. CHUẨN BỊ DATAFRAME TẬP TEST ---
print("\n--- Bước 2: Chuẩn bị dữ liệu test ---")
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'filename': test_filenames})
test_df['id'] = test_df['filename'].apply(lambda x: int(x.split('.')[0]))
test_df = test_df.sort_values(by='id').reset_index(drop=True)
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(TEST_DIR, x))
print(f"Tìm thấy {len(test_df)} ảnh trong tập test.")

# --- 3. TẠO DATASET VÀ DATALOADER CHO TẬP TEST ---
# Transforms cho TTA không chứa augmentation, vì ta sẽ áp dụng thủ công
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

test_dataset = TestImageDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoader cho tập test đã sẵn sàng.")

# --- 4. TẢI MODEL ĐÃ HUẤN LUYỆN ---
print(f"\n--- Bước 3: Tải model SWA từ '{MODEL_PATH}' ---")
model = models.get_model(MODEL_ARCH, weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4, inplace=True),
    torch.nn.Linear(num_ftrs, NUM_CLASSES)
)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Tải trọng số thành công!")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{MODEL_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    raise
model = model.to(DEVICE)
model.eval() # Bật chế độ evaluation

# --- 5. THỰC HIỆN DỰ ĐOÁN VỚI TTA ---
print("\n--- Bước 4: Bắt đầu dự đoán trên tập test với TTA ---")
all_preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Dự đoán với TTA"):
        images = images.to(DEVICE)
        
        # 1. Dự đoán trên ảnh gốc
        outputs_original = model(images)
        
        # 2. Dự đoán trên ảnh đã lật ngang
        # torch.flip lật tensor, dims=[3] tương ứng với chiều rộng (W) trong [N, C, H, W]
        outputs_flipped = model(torch.flip(images, dims=[3]))
        
        # 3. Lấy trung bình kết quả
        # Chuyển đổi logits sang xác suất bằng softmax trước khi lấy trung bình
        # để kết quả ổn định hơn.
        avg_probs = (torch.softmax(outputs_original, dim=1) + torch.softmax(outputs_flipped, dim=1)) / 2
        
        # 4. Lấy nhãn dự đoán từ xác suất trung bình cao nhất
        _, predicted = torch.max(avg_probs.data, 1)
        all_preds.extend(predicted.cpu().numpy())

# --- 6. TẠO FILE SUBMISSION ---
print("\n--- Bước 5: Tạo file submission.csv ---")
test_df['predicted_index'] = all_preds
test_df['label'] = test_df['predicted_index'] + 1 # Ánh xạ ngược 0,1,2,3 -> 1,2,3,4
submission_df = test_df[['id', 'label']]
submission_df.to_csv('submission_tta.csv', index=False)

print("\nĐã tạo file submission_tta.csv thành công!")
print("Xem trước 5 dòng đầu tiên:")
print(submission_df.head())


# ===================================================================
# ===== FULL PIPELINE: 5-FOLD CV & ENSEMBLE SUBMISSION ============
# ===================================================================
# Strategy:
# 1. Load a powerful pre-trained SWA model as a starting point.
# 2. Split the full training data into 5 Folds.
# 3. For each Fold:
#    a. Fine-tune the SWA model on the Fold's training data.
#    b. Apply SWA again to get the final robust model for the Fold.
# 4. For submission:
#    a. Load all 5 Fold-specific models.
#    b. Predict on the test set with all models.
#    c. Average their predictions (ensemble) to get the final result.
# ===================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, update_bn
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# PART 0: GLOBAL CONFIGURATION
# ==========================================================

# --- Main Configuration ---
# TODO: Make sure to load your full training dataframe 'df' here
# For example: df = pd.read_csv('/kaggle/input/modia-ml-2024/train.csv')
# df['filepath'] = df['id'].apply(lambda x: f'/kaggle/input/modia-ml-2024/train/train/{x}.png')
# label_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
# df['label_id'] = df['label'].map(label_mapping)

# --- Model & Image Config ---
IMG_SIZE = 300
MODEL_ARCH = 'efficientnet_b3'
BASE_SWA_MODEL_PATH = '/kaggle/working/swa_model_final.pth' # Your powerful starting point

# --- Training Config ---
BATCH_SIZE = 16 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4 # df['label_id'].nunique()
N_SPLITS = 5

# --- Epochs & Learning Rates Config (tuned for fine-tuning) ---
FINETUNE_EPOCHS_PER_FOLD = 15
SWA_EPOCHS_PER_FOLD = 10
LR_FN = 2e-5
SWA_LR = 5e-6

# --- Paths ---
TEST_DIR = '/kaggle/input/modia-ml-2024/test/test/'

# ==========================================================
# PART 1: HELPER CLASSES AND FUNCTIONS
# ==========================================================

# --- Dataset Classes ---
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label_id']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# --- Data Transforms ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    normalize,
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# --- Reusable Training Loop Function ---
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_path, val_dataset_len):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_acc = corrects.double() / val_dataset_len
        
        lr_to_print = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch+1} | Val Acc: {epoch_acc:.4f} | LR: {lr_to_print:.1e}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), model_path)
            print(f"*** New best model saved to '{model_path}' with accuracy: {best_acc:.4f} ***")

# ==========================================================
# PART 2: K-FOLD CROSS-VALIDATION TRAINING
# ==========================================================
print(f"Starting {N_SPLITS}-Fold CV Training. Base model: '{BASE_SWA_MODEL_PATH}'")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_id'])):
    print(f"\n{'='*25} FOLD {fold+1}/{N_SPLITS} {'='*25}")

    # --- Data preparation for the current fold ---
    train_df_fold = df.loc[train_idx].reset_index(drop=True)
    val_df_fold = df.loc[val_idx].reset_index(drop=True)
    train_dataset_fold = CustomImageDataset(train_df_fold, transform=train_transforms)
    val_dataset_fold = CustomImageDataset(val_df_fold, transform=val_transforms)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # --- Model paths for the current fold ---
    model_path_fold = f'best_model_b3_fold_{fold+1}.pth'
    swa_model_path_fold = f'swa_model_fold_{fold+1}.pth'
    
    # --- Initialize model and load base SWA weights ---
    model = models.get_model(MODEL_ARCH, weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(num_ftrs, NUM_CLASSES))
    
    print(f"--- [Fold {fold+1}] Loading base weights from '{BASE_SWA_MODEL_PATH}' ---")
    model.load_state_dict(torch.load(BASE_SWA_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # --- Stage 1: Fine-tuning on the fold's data ---
    print(f"\n--- [Fold {fold+1}] Stage 1: Fine-tuning ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer_fn = optim.AdamW(model.parameters(), lr=LR_FN, weight_decay=1e-3)
    scheduler_fn = CosineAnnealingLR(optimizer_fn, T_max=FINETUNE_EPOCHS_PER_FOLD, eta_min=1e-7)
    train_loop(model, train_loader_fold, val_loader_fold, criterion, optimizer_fn, 
               scheduler=scheduler_fn, num_epochs=FINETUNE_EPOCHS_PER_FOLD, 
               model_path=model_path_fold, val_dataset_len=len(val_dataset_fold))

    # --- Stage 2: Applying SWA ---
    print(f"\n--- [Fold {fold+1}] Stage 2: SWA ---")
    model.load_state_dict(torch.load(model_path_fold))
    optimizer_swa = optim.AdamW(model.parameters(), lr=SWA_LR)
    swa_model = AveragedModel(model)
    
    for epoch in range(SWA_EPOCHS_PER_FOLD):
        model.train()
        for inputs, labels in tqdm(train_loader_fold, desc=f"SWA Epoch {epoch+1}/{SWA_EPOCHS_PER_FOLD}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer_swa.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_swa.step()
        swa_model.update_parameters(model)
    
    print("Updating SWA batch normalization statistics...")
    update_bn(train_loader_fold, swa_model, device=DEVICE)
    torch.save(swa_model.module.state_dict(), swa_model_path_fold)
    print(f"*** SWA model for Fold {fold+1} saved to '{swa_model_path_fold}' ***")

print(f"\n{'='*20} FULL K-FOLD TRAINING COMPLETE {'='*20}")

# ==========================================================
# PART 3: ENSEMBLE PREDICTION & SUBMISSION
# ==========================================================
print("\n--- Starting final submission generation using ensemble of 5 models ---")

# --- Test data preparation ---
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({'filename': test_filenames})
test_df['id'] = test_df['filename'].apply(lambda x: int(x.split('.')[0]))
test_df = test_df.sort_values(by='id').reset_index(drop=True)
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(TEST_DIR, x))

test_dataset = TestImageDataset(test_df, transform=val_transforms) # Use val_transforms for test
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# --- Load all fold models ---
MODEL_PATHS = [f'swa_model_fold_{i+1}.pth' for i in range(N_SPLITS)]
models_list = []
for path in MODEL_PATHS:
    model = models.get_model(MODEL_ARCH, weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4, inplace=True),
        torch.nn.Linear(num_ftrs, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    models_list.append(model)
    print(f"Loaded model from: {path}")

# --- Run ensemble inference ---
all_preds = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Ensemble Predicting"):
        images = images.to(DEVICE)
        
        all_model_probs = []
        for model in models_list:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_model_probs.append(probs)
        
        avg_probs = torch.stack(all_model_probs).mean(dim=0)
        _, predicted = torch.max(avg_probs.data, 1)
        all_preds.extend(predicted.cpu().numpy())

# --- Create and save submission file ---
submission_df = pd.DataFrame({'id': test_df['id']})
submission_df['label'] = [p + 1 for p in all_preds] # Map 0,1,2,3 back to 1,2,3,4
submission_df.to_csv('submission_5fold_ensemble.csv', index=False)

print("\nSubmission file 'submission_5fold_ensemble.csv' created successfully!")
print("Top 5 rows:")
print(submission_df.head())

