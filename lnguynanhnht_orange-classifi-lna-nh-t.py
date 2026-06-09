# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import thư viện
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np



data_dir = '/kaggle/input/ai-training-challenge-hutech-orange-classifier/old_oranges_data_1/old_oranges_data/'

train_dir = os.path.join(data_dir, 'train_set')
test_dir = os.path.join(data_dir, 'test_set')


import os
import numpy as np
import pandas as pd
from PIL import Image


# Hàm đọc dữ liệu và lưu trữ vào dataframe
def load_data_to_dataframe(data_dir):
    data = []
    
    # Duyệt qua các thư mục con "Orange_Bad" và "Orange_Good"
    for label, subdir in enumerate(["Orange_Bad", "Orange_Good"]):
        subdir_path = os.path.join(data_dir, subdir)
        
        # Duyệt qua các ảnh trong mỗi thư mục con
        for filename in os.listdir(subdir_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Kiểm tra đuôi tệp
                img_path = os.path.join(subdir_path, filename)
                
                # Tải ảnh và thay đổi kích thước về (224, 224)
                img = Image.open(img_path).resize((224, 224))
                
                # Chuyển ảnh thành mảng numpy
                img_array = np.array(img)
                
                # Chuyển thành một danh sách một chiều
                img_flatten = img_array.flatten()
                
                # Gán nhãn: 0 cho "Orange_Good", 1 cho "Orange_Bad"
                label_value = 1 if subdir == "Orange_Bad" else 0
                
                # Thêm vào danh sách
                data.append([filename, img_flatten, label_value])
    
    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(data, columns=["id", "values", "label"])
    
    return df

# Đọc dữ liệu train và test và lưu vào dataframe
train_df = load_data_to_dataframe(train_dir)
test_df = load_data_to_dataframe(test_dir)





train_df.head()


import matplotlib.pyplot as plt

# Vẽ 5 bức ảnh đầu tiên của train_df với cột 'values'
plt.figure(figsize=(10, 10))
df_temp  =train_df[0:6]
for i in range(5):
    img_array = df_temp.iloc[i]['values'].reshape(224, 224, 3)  # Chuyển mảng 1 chiều về hình ảnh 224x224
    plt.subplot(1, 5, i + 1)
    plt.imshow(img_array.astype('uint8'))
    plt.title(f"Image {i+1}")
    plt.axis('off')

plt.show()



import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def preprocess_image(img_array):
    # Bước 1: Tăng cường hình ảnh (Image Enhancement)
    # Tăng độ sáng
    img = Image.fromarray(img_array)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.2)  # Tăng sáng lên 20%
    
    # Tăng độ tương phản
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Tăng độ tương phản lên 50%

    # Bước 2: Lọc median (Median Filtering)
    img_array = np.array(img)
    filtered_img = cv2.medianBlur(img_array, 5)  # Lọc median với kernel size = 5

    # Bước 3: Biến đổi không gian màu (Color Space Transformation)
    # Chuyển đổi không gian màu từ BGR sang HSV
    hsv_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2HSV)

    # Bước 4: Loại bỏ nhiễu (Noise Removing)
    # Sử dụng Gaussian Blur để loại bỏ nhiễu
    noise_removed_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)

    # Bước 5: Cân bằng histogram (Histogram Equalization)
    # Bỏ qua chuyển ảnh về gray scale và trực tiếp thực hiện cân bằng histogram trên ảnh HSV
    equalized_img = cv2.equalizeHist(noise_removed_img[:, :, 2])  # Cân bằng histogram chỉ trên kênh V (Value)

    # Gắn lại vào kênh V của ảnh HSV
    noise_removed_img[:, :, 2] = equalized_img

    # Chuyển lại ảnh từ HSV về BGR để tiếp tục xử lý
    final_img = cv2.cvtColor(noise_removed_img, cv2.COLOR_HSV2BGR)

    return final_img

# Hàm xử lý các ảnh trong dataframe và lưu vào cột 'values_process'
def process_dataframe(df):
    processed_images = []
    
    for index, row in df.iterrows():
        # Đọc mảng pixel từ cột 'values'
        img_array = np.array(row['values'], dtype=np.uint8).reshape(224, 224, 3)
        
        # Tiến hành tiền xử lý cho mỗi ảnh
        processed_img = preprocess_image(img_array)
        
        # Thêm ảnh đã xử lý vào danh sách
        processed_images.append(processed_img)
    
    # Lưu kết quả vào cột 'values_process'
    df['values_process'] = processed_images
    
    return df

# Giả sử train_df là DataFrame của bạn chứa các ảnh và nhãn
# Xử lý dữ liệu và lưu vào cột 'values_process'
train_df = process_dataframe(train_df)

# Hiển thị ảnh đã xử lý từ cột 'values_process'
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_df['values_process'].iloc[i])
    plt.title(f"Image {i+1}")
    plt.axis('off')

plt.show()



test_df = process_dataframe(test_df)


test_df.head()


train_df.head()


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def apply_edge_detection_with_color(img_array):
    # Chuyển ảnh sang ảnh xám
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng Canny Edge Detection
    edges = cv2.Canny(gray_img, 100, 200)  # Các tham số 100 và 200 là ngưỡng cho Canny
    
    # Áp dụng bảng màu 'JET' để tạo màu sắc cho ảnh biên
    colored_edges = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    
    return colored_edges

# Hàm xử lý Edge Detection với màu sắc cho các ảnh trong dataframe và lưu vào cột 'values_edge_color'
def process_edge_detection_with_color_on_dataframe(df):
    edge_images_colored = []
    
    for index, row in df.iterrows():
        # Đọc mảng pixel từ cột 'values' và reshape thành 224x224x3
        img_array = np.array(row['values'], dtype=np.uint8).reshape(224, 224, 3)
        
        # Áp dụng Edge Detection với màu sắc cho ảnh
        edge_image_colored = apply_edge_detection_with_color(img_array)
        
        # Thêm ảnh biên có màu vào danh sách
        edge_images_colored.append(edge_image_colored)
    
    # Lưu ảnh đã qua Edge Detection màu sắc vào cột 'values_edge_color'
    df['values_edge_color'] = edge_images_colored
    return df

# Giả sử train_df đã có cột 'values' với ảnh đã xử lý
train_df = process_edge_detection_with_color_on_dataframe(train_df)

# Hiển thị 5 ảnh đầu tiên từ cột 'values_edge_color' với màu sắc
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_df['values_edge_color'].iloc[i])  # Hiển thị ảnh với màu sắc
    plt.title(f"Edge Detected Image {i+1}")
    plt.axis('off')

plt.show()



train_df.head()


!pip install mahotas



import cv2
import numpy as np
import pandas as pd
import mahotas
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def extract_features_and_visualize(image):
    features = {}

    # **Texture Features using GLCM (Gray Level Co-occurrence Matrix) from mahotas**
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = mahotas.features.haralick(gray)  # Tính toán GLCM
    contrast = glcm[0, 1]  # Contrast
    homogeneity = glcm[0, 3]  # Homogeneity
    energy = glcm[0, 4]  # Energy
    correlation = glcm[0, 5]  # Correlation
    features['Texture'] = [contrast, homogeneity, energy, correlation]

    # **Color Features (mean of HSV channels)**
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])
    features['Color'] = [mean_hue, mean_saturation, mean_value]

    # **Shape Features using Contours (Area and Perimeter)**
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = cv2.contourArea(contours[0])
    contour_perimeter = cv2.arcLength(contours[0], True)
    features['Shape'] = [contour_area, contour_perimeter]

    # **ORB Features (Oriented FAST and Rotated BRIEF)**
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    features['ORB'] = len(keypoints)  # Store number of keypoints detected

    # **HOG Features (Histogram of Oriented Gradients)**
    hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    features['HOG'] = hog_features

    # **SIFT Features (Scale-Invariant Feature Transform)**
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(image, None)
    features['SIFT'] = len(keypoints_sift)  # Store number of keypoints detected

    # **Edge Detection (Canny)**
    edges = cv2.Canny(gray, 100, 200)
    features['Edge Detection'] = edges

    # Visualize Results
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Original Image
    ax[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title('Original Image')
    ax[0, 0].axis('off')

    # Texture Features Visualization
    ax[0, 1].text(0.5, 0.5, f"Texture Features\nContrast: {contrast:.2f}\nHomogeneity: {homogeneity:.2f}", fontsize=12,
                  ha='center')
    ax[0, 1].axis('off')

    # Color Features Visualization
    ax[0, 2].text(0.5, 0.5, f"Color Features\nHue: {mean_hue:.2f}\nSaturation: {mean_saturation:.2f}", fontsize=12,
                  ha='center')
    ax[0, 2].axis('off')

    # Shape Features Visualization
    ax[1, 0].text(0.5, 0.5, f"Shape Features\nArea: {contour_area:.2f}\nPerimeter: {contour_perimeter:.2f}", fontsize=12,
                  ha='center')
    ax[1, 0].axis('off')

    # HOG Features Visualization
    ax[1, 1].imshow(hog_image_rescaled, cmap='gray')
    ax[1, 1].set_title('HOG Features')
    ax[1, 1].axis('off')

    # Edge Detection Visualization
    ax[1, 2].imshow(edges, cmap='gray')
    ax[1, 2].set_title('Edge Detection (Canny)')
    ax[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    return features


# Example: Apply to one image in the dataset
# Assume `train_df` contains the 'values' column with images

image = np.array(train_df['values'].iloc[1], dtype=np.uint8).reshape(224, 224, 3)  # Get an example image
extracted_features = extract_features_and_visualize(image)

# You can access the features like this:
print("Extracted Features:", extracted_features)



# Tạo thư mục output nếu chưa có
output_dir = '/kaggle/working/augmented'
os.makedirs(output_dir, exist_ok=True)


import os

# Đường dẫn thư mục
output_dir = '/kaggle/working/augmented'

# Duyệt qua tất cả tệp trong thư mục và xóa chúng
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path):  # Kiểm tra nếu là tệp
        os.remove(file_path)  # Xóa tệp

print("Tất cả các tệp trong thư mục đã được xóa.")



import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# Tạo các phép biến đổi tăng cường dữ liệu và chuyển ảnh sang xám
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Chuyển ảnh sang màu xám (1 kênh)
    transforms.RandomRotation(30),  # Xoay ảnh ngẫu nhiên trong khoảng 30 độ
    transforms.RandomHorizontalFlip(),  # Lật ngang ảnh
    transforms.RandomVerticalFlip(),  # Lật dọc ảnh
    transforms.Resize((224, 224)),  # Đảm bảo kích thước ảnh là 224x224
    transforms.ToTensor(),  # Chuyển ảnh thành tensor
])

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lấy ảnh và nhãn từ DataFrame
        image_array = np.array(self.df['values_process'].iloc[idx], dtype=np.uint8)
        label = self.df['label'].iloc[idx]

        # Chuyển mảng numpy thành ảnh PIL
        img = Image.fromarray(image_array)

        # Áp dụng các phép biến đổi nếu có
        if self.transform:
            img = self.transform(img)

        return img, label

# Tạo Dataset từ train_df và test_df
train_dataset = CustomImageDataset(train_df, transform=transform)
test_dataset = CustomImageDataset(test_df, transform=transform)

# Chia train_dataset thành train và validation datasets
train_size = int(0.7 * len(train_dataset))  # 70% cho train
val_size = len(train_dataset) - train_size  # 30% cho validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Tạo DataLoader cho train, validation và test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Duyệt qua các batch và hiển thị các ảnh tăng cường cho train_loader
for images, labels in train_loader:
    print(images.shape)  # Kích thước của batch
    print(labels)  # Nhãn của batch
    break  # Chỉ in ra một batch đầu tiên

# Duyệt qua các batch và hiển thị các ảnh của validation set
for images, labels in val_loader:
    print(images.shape)  # Kích thước của batch
    print(labels)  # Nhãn của batch
    break  # Chỉ in ra một batch đầu tiên

# Duyệt qua các batch và hiển thị các ảnh của test set
for images, labels in test_loader:
    print(images.shape)  # Kích thước của batch
    print(labels)  # Nhãn của batch
    break  # Chỉ in ra một batch đầu tiên



train_label_0 = 0
train_label_1 = 0
for images, labels in train_loader:
    train_label_0 += (labels == 0).sum().item()  # Đếm nhãn 0
    train_label_1 += (labels == 1).sum().item()  # Đếm nhãn 1

print(f"Trong tập train: Nhãn 0 = {train_label_0}, Nhãn 1 = {train_label_1}")

# Đếm số lượng nhãn 0 và 1 trong val_loader
val_label_0 = 0
val_label_1 = 0
for images, labels in val_loader:
    val_label_0 += (labels == 0).sum().item()  # Đếm nhãn 0
    val_label_1 += (labels == 1).sum().item()  # Đếm nhãn 1

print(f"Trong tập validation: Nhãn 0 = {val_label_0}, Nhãn 1 = {val_label_1}")



# Tạo thư mục checkpoint trên Kaggle
checkpoint_dir = '/kaggle/working/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import pandas as pd

# Tạo thư mục checkpoint trên Kaggle
checkpoint_dir = '/kaggle/working/checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Lớp conv 1 với 1 kênh đầu vào (xám)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Lớp conv 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Lớp conv 3
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Lớp conv 4
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Lớp conv 5
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout sau mỗi lớp Conv
        self.dropout_conv = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(256 * 7 * 7, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 1)  # Lớp output (cam sạch hoặc cam hư)
        self.sigmoid = nn.Sigmoid()

        # Dropout sau lớp fully connected
        self.dropout_fc = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout_conv(x)  # Dropout sau conv1
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout_conv(x)  # Dropout sau conv2
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.dropout_conv(x)  # Dropout sau conv3
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.dropout_conv(x)  # Dropout sau conv4
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.dropout_conv(x)  # Dropout sau conv5
        
        x = x.view(-1, 256 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Dropout sau fc1
        x = self.sigmoid(self.fc2(x))
        return x

# Khởi tạo mô hình, loss function và optimizer
model = CNNModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Cài đặt Device (GPU hoặc CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Cài đặt EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path='checkpoint/best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_accuracy = 0  # Đánh giá bằng accuracy
        self.best_model_wts = None
        self.path = path

    def __call__(self, val_accuracy, model):
        # Nếu val_accuracy cao hơn, lưu lại mô hình
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model_wts = model.state_dict()
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                return True
        return False

# Khởi tạo EarlyStopping
early_stopping = EarlyStopping(patience=10, verbose=True, path=checkpoint_dir + '/best_model.pt')

# Huấn luyện mô hình
epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass và optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Tính accuracy
        predicted = (outputs.squeeze() > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracy = correct_train / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            running_val_loss += loss.item()

            # Tính accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # EarlyStopping
    if early_stopping(val_accuracy, model):
        print("Early stopping")
        break

# Vẽ quá trình huấn luyện và validation
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
checkpoint_path = '/kaggle/working/checkpoint/best_model.pt'
# Tải mô hình đã lưu
model = CNNModel()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# Chuyển mô hình và dữ liệu sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dự đoán trên test_loader và tính toán các chỉ số
y_true = []
y_pred = []

# Duyệt qua các batch trong test_loader
with torch.no_grad():  # Không cần tính toán gradients khi chỉ dự đoán
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Dự đoán nhãn (giới hạn ngưỡng ở 0.5 để phân loại nhị phân)
        predicted = (outputs.squeeze() > 0.5).float()

        # Lưu nhãn thực và nhãn dự đoán
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Chuyển y_true và y_pred thành numpy array
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Tính toán các chỉ số
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# In ra các chỉ số đánh giá
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


test_df.head()


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np

checkpoint_path = '/kaggle/working/checkpoint/best_model.pt'

# Tải mô hình đã lưu
model = CNNModel()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()  # Chuyển mô hình sang chế độ đánh giá

# Chuyển mô hình và dữ liệu sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dự đoán trên test_loader và tính toán các chỉ số
y_true = []
y_pred = []

# Duyệt qua các batch trong test_loader và dự đoán
predicts = []  # Danh sách lưu các dự đoán
with torch.no_grad():  # Không cần tính toán gradients khi chỉ dự đoán
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Dự đoán nhãn (giới hạn ngưỡng ở 0.5 để phân loại nhị phân)
        predicted = (outputs.squeeze() > 0.5).float()

        # Lưu nhãn thực và nhãn dự đoán
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        # Lưu kết quả dự đoán vào list
        predicts.extend(predicted.cpu().numpy())

# Cập nhật cột 'predict' trong test_df với các dự đoán
test_df['predict'] = predicts

# Tính toán các chỉ số
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# In ra một số kết quả trong test_df để xác nhận
print(test_df.head())



# Lấy các cột 'id' và 'predict' từ test_df
filenames = test_df['id']
predictions = test_df['predict']

# Tạo DataFrame mới để lưu kết quả
results = pd.DataFrame({
    'image_name': filenames,  # Tên ảnh từ cột 'id'
    'label': predictions      # Dự đoán nhãn từ cột 'predict'
})

# In kết quả
print(results.head())



results


import pandas as pd

def read_filecsv(file_path):
    """
    Đọc file CSV và trả về một DataFrame.
    
    Parameters:
    file_path (str): Đường dẫn tới tệp CSV cần đọc.

    Returns:
    pd.DataFrame: DataFrame chứa dữ liệu từ file CSV.
    """
    # Đọc file CSV và trả về DataFrame
    df = pd.read_csv(file_path, sep=',')
    return df

# Ví dụ gọi hàm
file_path = '/kaggle/working/results.csv'  # Đường dẫn tới tệp CSV
results = read_filecsv(file_path)

# Bước 1: Loại bỏ phần mở rộng .jpg hoặc .png
results['image_name'] = results['image_name'].str.replace(r'.jpg|.png', '', regex=True)

# Bước 2: Tách ra phần trước và sau dấu _ (phần `black` hoặc `fresh` và số)
results['prefix'] = results['image_name'].str.extract(r'(black|fresh)')
results['number'] = results['image_name'].str.extract(r'(\d+)$').astype(int)

# Bước 3: Sắp xếp theo prefix (black trước rồi đến fresh), sau đó theo số giảm dần
results_sorted = results.sort_values(by=['prefix', 'number'], ascending=[True, True])

# Bước 4: Loại bỏ cột phụ và lưu kết quả
results_sorted = results_sorted.drop(columns=['prefix', 'number'])

# In kết quả đã sắp xếp
print(results_sorted)








# Lưu kết quả vào file CSV
results_sorted.to_csv('/kaggle/working/results.csv', sep=',', index=False)



