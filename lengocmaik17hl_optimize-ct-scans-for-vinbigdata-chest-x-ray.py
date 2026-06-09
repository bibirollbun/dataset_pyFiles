# Install specific versions of required packages
!pip install numpy pandas matplotlib tqdm torch torchvision scikit-learn scipy pillow

# Ensure tqdm is compatible with Jupyter/Kaggle notebooks
!pip install tqdm --upgrade

!pip install lime shap -qqq


!git clone https://github.com/jacobgil/pytorch-grad-cam.git
%cd pytorch-grad-cam
!pip install .
%cd ..


import pandas as pd
import numpy as np
import os
import shutil

# Đường dẫn đến dữ liệu gốc trên Kaggle
KAGGLE_INPUT_PATH = '/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/'
TRAIN_CSV_PATH = os.path.join(KAGGLE_INPUT_PATH, 'train.csv')
TRAIN_IMAGES_PATH = os.path.join(KAGGLE_INPUT_PATH, 'train/')

# Đường dẫn thư mục đầu ra
OUTPUT_BASE_PATH = '/kaggle/working/'
ABNORMAL_PATH = os.path.join(OUTPUT_BASE_PATH, 'abnormal/')
NON_ABNORMAL_PATH = os.path.join(OUTPUT_BASE_PATH, 'non_abnormal/')

# Số lượng ảnh tối đa muốn lấy cho mỗi loại
MAX_IMAGES_PER_CATEGORY = 500

# Tạo các thư mục đầu ra nếu chúng chưa tồn tại
os.makedirs(ABNORMAL_PATH, exist_ok=True)
os.makedirs(NON_ABNORMAL_PATH, exist_ok=True)

print(f"Đọc file CSV từ: {TRAIN_CSV_PATH}")
# Đọc file train.csv
df = pd.read_csv(TRAIN_CSV_PATH)
print(f"Đã đọc {len(df)} dòng từ train.csv.")

# Lấy tất cả các image_id duy nhất
all_image_ids = df['image_id'].unique()
print(f"Tổng số ảnh duy nhất: {len(all_image_ids)}")

# Xác định image_id của các ảnh "không bệnh" (chỉ có class_id = 14)
# Group theo image_id và kiểm tra xem có bất kỳ class_id nào khác 14 không
image_labels = df.groupby('image_id')['class_id'].apply(list).reset_index()

non_abnormal_image_ids = []
abnormal_image_ids = []

for index, row in image_labels.iterrows():
    image_id = row['image_id']
    class_ids = row['class_id']
    
    # Nếu tất cả các nhãn cho ảnh này đều là 14 ("No finding")
    if all(cid == 14 for cid in class_ids):
        non_abnormal_image_ids.append(image_id)
    else:
        abnormal_image_ids.append(image_id)

print(f"Tổng số ảnh 'không bệnh' (non_abnormal) tìm thấy: {len(non_abnormal_image_ids)}")
print(f"Tổng số ảnh 'có bệnh' (abnormal) tìm thấy: {len(abnormal_image_ids)}")

print(f"\nBắt đầu sao chép tối đa {MAX_IMAGES_PER_CATEGORY} ảnh cho mỗi loại...")

# Sao chép ảnh "không bệnh"
copied_non_abnormal_count = 0
for image_id in non_abnormal_image_ids:
    if copied_non_abnormal_count >= MAX_IMAGES_PER_CATEGORY:
        break # Dừng lại nếu đã đủ số lượng
    
    src_path = os.path.join(TRAIN_IMAGES_PATH, f"{image_id}.dicom")
    dst_path = os.path.join(NON_ABNORMAL_PATH, f"{image_id}.dicom")
    try:
        shutil.copy(src_path, dst_path)
        copied_non_abnormal_count += 1
    except FileNotFoundError:
        # print(f"Cảnh báo: Không tìm thấy file {src_path}. Bỏ qua.")
        pass # Bỏ qua cảnh báo để tránh spam console
    except Exception as e:
        print(f"Lỗi khi sao chép {src_path} đến {dst_path}: {e}")

print(f"Đã sao chép {copied_non_abnormal_count} ảnh vào thư mục non_abnormal.")

# Sao chép ảnh "có bệnh"
copied_abnormal_count = 0
for image_id in abnormal_image_ids:
    if copied_abnormal_count >= MAX_IMAGES_PER_CATEGORY:
        break # Dừng lại nếu đã đủ số lượng
    
    src_path = os.path.join(TRAIN_IMAGES_PATH, f"{image_id}.dicom")
    dst_path = os.path.join(ABNORMAL_PATH, f"{image_id}.dicom")
    try:
        shutil.copy(src_path, dst_path)
        copied_abnormal_count += 1
    except FileNotFoundError:
        # print(f"Cảnh báo: Không tìm thấy file {src_path}. Bỏ qua.")
        pass # Bỏ qua cảnh báo để tránh spam console
    except Exception as e:
        print(f"Lỗi khi sao chép {src_path} đến {dst_path}: {e}")

print(f"Đã sao chép {copied_abnormal_count} ảnh vào thư mục abnormal.")

print("\nQuá trình phân chia dữ liệu hoàn tất!")
print(f"Kiểm tra các thư mục: {ABNORMAL_PATH} và {NON_ABNORMAL_PATH}")


import os
import pydicom
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm

# Đường dẫn đến các thư mục đã phân chia (từ bước trước)
ABNORMAL_DICOM_PATH = '/kaggle/working/abnormal/'
NON_ABNORMAL_DICOM_PATH = '/kaggle/working/non_abnormal/'

# Đường dẫn mới để lưu ảnh đã chuyển đổi
ABNORMAL_PNG_PATH = '/kaggle/working/abnormal_png/'
NON_ABNORMAL_PNG_PATH = '/kaggle/working/non_abnormal_png/'

# Tạo các thư mục mới nếu chúng chưa tồn tại
os.makedirs(ABNORMAL_PNG_PATH, exist_ok=True)
os.makedirs(NON_ABNORMAL_PNG_PATH, exist_ok=True)

def dicom_to_png(dicom_dir, png_dir):
    """
    Chuyển đổi tất cả các file DICOM trong một thư mục sang định dạng PNG.
    """
    print(f"Bắt đầu chuyển đổi DICOM từ '{dicom_dir}' sang PNG trong '{png_dir}'...")
    dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dicom')]
    
    for dicom_file in tqdm(dicom_files, desc=f"Chuyển đổi từ {os.path.basename(dicom_dir)}"):
        dicom_path = os.path.join(dicom_dir, dicom_file)
        png_filename = dicom_file.replace('.dicom', '.png')
        png_path = os.path.join(png_dir, png_filename)
        
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            
            # Kiểm tra xem pixel_array có sẵn không
            if 'PixelData' in dicom_data:
                # Chuyển đổi dữ liệu pixel sang mảng numpy
                pixel_array = dicom_data.pixel_array
                
                # Chuẩn hóa giá trị pixel về 0-255 (nếu cần) và chuyển sang uint8
                # Một số hình ảnh DICOM có thể có độ sâu bit cao hơn
                if pixel_array.dtype != np.uint8:
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
                    pixel_array = (pixel_array * 255).astype(np.uint8)
                
                # Nếu là ảnh grayscale 2D, chuyển thành RGB để tương thích với các mô hình CNN
                if len(pixel_array.shape) == 2:
                    image = Image.fromarray(pixel_array).convert('RGB')
                elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 1: # Grayscale với 3rd dim
                     image = Image.fromarray(pixel_array[:,:,0]).convert('RGB')
                else: # Đã là RGB hoặc có nhiều kênh
                    image = Image.fromarray(pixel_array)

                image.save(png_path)
            else:
                print(f"Cảnh báo: File '{dicom_file}' không chứa PixelData.")
        except Exception as e:
            print(f"Lỗi khi xử lý file '{dicom_file}': {e}")

# Chạy chuyển đổi
dicom_to_png(ABNORMAL_DICOM_PATH, ABNORMAL_PNG_PATH)
dicom_to_png(NON_ABNORMAL_DICOM_PATH, NON_ABNORMAL_PNG_PATH)

print("\nĐã hoàn tất chuyển đổi DICOM sang PNG.")
print(f"Ảnh abnormal đã chuyển đổi được lưu tại: {ABNORMAL_PNG_PATH}")
print(f"Ảnh non_abnormal đã chuyển đổi được lưu tại: {NON_ABNORMAL_PNG_PATH}")


import os
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # For nice progress bars in Kaggle notebooks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix, brier_score_loss
from scipy.special import softmax # For converting logits to probabilities
from sklearn.metrics.pairwise import cosine_similarity # For similarity-based confidence adjustment

# --- XAI Library Imports ---
from pytorch_grad_cam import GradCAMPlusPlus # Using GradCAMPlusPlus as requested
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Removed LIME and SHAP imports as requested.
# import lime
# import lime.lime_image
# import shap

# --- Configuration and Hyperparameters ---
class Config:
    """
    Cấu hình và siêu tham số cho toàn bộ quy trình phân loại có chọn lọc.
    Các tham số này có thể được điều chỉnh để phù hợp với các tập dữ liệu
    và yêu cầu khác nhau.
    """
    def __init__(self):
        # Đường dẫn tập dữ liệu (giả sử môi trường Kaggle)
        self.DATA_DIR = '/kaggle/working/' # Giữ nguyên hoặc điều chỉnh nếu bạn muốn một thư mục gốc khác
        self.ABNORMAL_DIR = os.path.join(self.DATA_DIR, '/kaggle/working/abnormal_png') # Đã chuyển đổi
        self.NON_ABNORMAL_DIR = os.path.join(self.DATA_DIR, '/kaggle/working/non_abnormal_png') # Đã chuyển đổi
        
        # Mô hình và Huấn luyện
        self.IMAGE_SIZE = (224, 224) # Kích thước chuẩn cho nhiều CNN tiền huấn luyện
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS_PER_MODEL = 2 # Tăng số epoch để huấn luyện tốt hơn
        self.LEARNING_RATE = 1e-4
        self.NUM_ENSEMBLE_MODELS = 2 # Số lượng mô hình trong ensemble (Thay đổi từ 1 thành 3 để dynamic gating có ý nghĩa)

        # Cấu hình Monte Carlo Dropout (MCDO)
        self.MCDO_ENABLE = False      # Bật/tắt Monte Carlo Dropout
        self.MCDO_DROPOUT_RATE = 0.3   # Tỷ lệ dropout cho MCDO
        self.MCDO_NUM_RUNS = 10        # Số lần chạy forward pass cho MCDO để ước tính độ bất định

        # Mục tiêu từ chối
        self.TARGET_ACCEPTED_ACCURACY = 0.99 # 99.5% độ chính xác trên các trường hợp được chấp nhận
        self.TARGET_REJECTION_RATE = 0.07      # Từ chối khoảng 10% các trường hợp

        # Các tham số có thể điều chỉnh thủ công cho phân loại chọn lọc (thử nghiệm với chúng!)
        self.DISAGREEMENT_PENALTY_FACTOR = 5.0 # Mức độ bất đồng của ensemble làm giảm độ tin cậy
        self.TRAINING_DYNAMICS_CONF_PENALTY = 0.1 # Mức độ các ví dụ "khó học" trong quá trình huấn luyện làm giảm độ tin cậy của các ví dụ test tương tự
        
        # Trọng số cho mục tiêu tối ưu hóa ngưỡng từ chối (mức độ quan trọng của mỗi yếu tố)
        # Các trọng số này cho phép điều chỉnh sự đánh đổi giữa độ chính xác được chấp nhận, tỷ lệ từ chối và ECE
        self.ACCURACY_DEVIATION_WEIGHT = 10.0 # Trọng số cao để mạnh mẽ thực thi TARGET_ACCEPTED_ACCURACY
        self.REJECTION_RATE_DEVIATION_WEIGHT = 1.0 # Trọng số chuẩn cho độ lệch tỷ lệ từ chối
        self.ECE_DEVIATION_WEIGHT = 5.0       # Trọng số cho ECE trong tối ưu hóa ngưỡng từ chối (điều chỉnh cái này, giá trị cao hơn có nghĩa là tập chấp nhận được hiệu chỉnh tốt hơn)

        # Ngưỡng phát hiện OOD (có thể điều chỉnh thủ công)
        self.OOD_CONFIDENCE_THRESHOLD = 0.65 # Các mẫu dưới độ tin cậy này
        self.OOD_VARIANCE_THRESHOLD = 0.08  # Và trên phương sai này là OOD tiềm năng
        # Bật/tắt Weighted Logits theo Độ tin cậy (ECE)
        self.USE_WEIGHTED_ENSEMBLE = True 
        
        # Bật/tắt Dynamic Ensemble Selection (chỉ chọn model tốt nhất cho từng mẫu)
        # Lưu ý: Đặt NUM_ENSEMBLE_MODELS = 3 để logic chọn 2/3 hoạt động đúng
        self.USE_DYNAMIC_SELECTION = True
        self.DYNAMIC_SELECTION_COUNT = 2 # Chọn 2 model tốt nhất

        # Hằng số nhỏ để tránh chia cho 0 khi tính trọng số từ ECE
        self.EPSILON_ECE = 1e-8
        # Các cài đặt khác
        self.RANDOM_SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_SAVE_DIR = '/kaggle/working/models' # Thư mục để lưu các mô hình ensemble đã huấn luyện
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        self.XAI_SAVE_DIR = '/kaggle/working/xai_visualizations' # Thư mục để lưu các hình ảnh XAI
        os.makedirs(self.XAI_SAVE_DIR, exist_ok=True) # Đảm bảo thư mục đầu ra XAI tồn tại

        # Temperature Scaling
        self.ENABLE_TEMPERATURE_SCALING = True
        self.TEMPERATURE_CALIBRATION_SPLIT = 0.1 # Tỷ lệ dữ liệu test được dùng để hiệu chỉnh nhiệt độ
        self.CALIBRATION_EPOCHS = 20 # Số epoch cho huấn luyện hiệu chỉnh nhiệt độ
        self.CALIBRATION_LR = 0.01 # Tốc độ học cho hiệu chỉnh nhiệt độ

        # Training Dynamics Analysis
        self.ENABLE_TRAINING_DYNAMICS = True # Bật/tắt phân tích động lực học huấn luyện

cfg = Config()

# Thiết lập seed ngẫu nhiên để tái sản xuất kết quả
def set_seed(seed):
    """Đặt seed ngẫu nhiên để tái sản xuất kết quả trên các thư viện khác nhau."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.RANDOM_SEED)

print(f"Sử dụng thiết bị: {cfg.DEVICE}")

# --- 1. Tải và tiền xử lý dữ liệu ---
class CTScanDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh để tải hình ảnh CT scan và nhãn của chúng.
    Xử lý đường dẫn hình ảnh và áp dụng các phép biến đổi.
    Trả về hình ảnh, nhãn, và chỉ mục toàn cục gốc của nó.
    """
    def __init__(self, image_paths, labels, transform=None, global_indices=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        # Đảm bảo global_indices là một mảng numpy
        self.global_indices = np.array(global_indices) if global_indices is not None else np.arange(len(image_paths))
        # Tạo ánh xạ từ global_idx đến local idx trong phân tách này để tra cứu thuận tiện
        self.original_indices_map = {self.global_indices[i]: i for i in range(len(self.global_indices))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Đảm bảo 3 kênh cho các mô hình tiền huấn luyện
        label = self.labels[idx]
        global_idx = self.global_indices[idx] # Trả về chỉ mục toàn cục gốc

        if self.transform:
            image = self.transform(image)

        return image, label, global_idx # Trả về chỉ mục toàn cục gốc để theo dõi động lực huấn luyện


def prepare_datasets(cfg):
    """
    Thu thập đường dẫn hình ảnh và nhãn, sau đó chia thành các tập huấn luyện, validation và test.
    Gán một chỉ mục toàn cục duy nhất cho mỗi hình ảnh từ tập dữ liệu gốc.
    """
    all_image_paths_raw = []
    all_labels_raw = []

    # Thu thập hình ảnh Abnormal
    abnormal_paths = [os.path.join(cfg.ABNORMAL_DIR, f) for f in os.listdir(cfg.ABNORMAL_DIR) if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(abnormal_paths)
    all_labels_raw.extend([1] * len(abnormal_paths)) # 1 cho 'có bệnh'
    
    # Thu thập hình ảnh Non-Abnormal
    non_abnormal_paths = [os.path.join(cfg.NON_ABNORMAL_DIR, f) for f in os.listdir(cfg.NON_ABNORMAL_DIR) if f.endswith('.png') or f.endswith('.jpg')]
    all_image_paths_raw.extend(non_abnormal_paths)
    all_labels_raw.extend([0] * len(non_abnormal_paths)) # 0 cho 'không bệnh'

    # Gán chỉ mục toàn cục
    all_global_indices = list(range(len(all_image_paths_raw)))

    print(f"Tổng số hình ảnh tìm thấy: {len(all_image_paths_raw)}")
    print(f"Hình ảnh COVID: {len(abnormal_paths)}, Hình ảnh Non-COVID: {len(non_abnormal_paths)}")

    # Tạo các phân tách phân tầng cho tập huấn luyện+validation và tập test trước
    # Mong muốn: Test = 15% tổng số
    train_val_paths, test_paths, train_val_labels, test_labels, \
    train_val_global_indices, test_global_indices = train_test_split(
        all_image_paths_raw, all_labels_raw, all_global_indices,
        test_size=0.15, random_state=cfg.RANDOM_SEED, stratify=all_labels_raw
    )
    
    # Sau đó, chia tập train_val thành các tập huấn luyện và validation thực tế
    # train_val là 85% tổng số. Chúng ta muốn Val = 15% tổng số.
    # Vì vậy, val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    val_size_relative_to_train_val = 0.15 / (1.0 - 0.15)
    train_paths, val_paths, train_labels, val_labels, \
    train_global_indices, val_global_indices = train_test_split(
        train_val_paths, train_val_labels, train_val_global_indices,
        test_size=val_size_relative_to_train_val, random_state=cfg.RANDOM_SEED, stratify=train_val_labels
    )

    print(f"Kích thước tập huấn luyện: {len(train_paths)}")
    print(f"Kích thước tập Validation: {len(val_paths)}")
    print(f"Kích thước tập Test: {len(test_paths)}") # test_paths này tham chiếu đến tập test cuối cùng, độc lập

    # Định nghĩa các phép biến đổi
    train_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa ImageNet
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Truyền global_indices vào constructor của Dataset
    train_dataset = CTScanDataset(train_paths, train_labels, train_transform, train_global_indices)
    val_dataset = CTScanDataset(val_paths, val_labels, val_test_transform, val_global_indices)
    test_dataset = CTScanDataset(test_paths, test_labels, val_test_transform, test_global_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# --- 2. Định nghĩa Base Classifier ---
class BaseClassifier(nn.Module):
    """
    Một bộ phân loại CNN cơ bản sử dụng mô hình ResNet tiền huấn luyện.
    Bao gồm một bộ trích xuất đặc trưng rõ ràng để dễ dàng kết nối cho Grad-CAM
    và để lấy đặc trưng cho phân tích dựa trên sự tương đồng.
    Tích hợp các lớp Dropout cho Monte Carlo Dropout (MCDO).
    """
    def __init__(self, num_classes=2, dropout_rate=0.0):
        super(BaseClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        # Tải mô hình ResNet18 tiền huấn luyện
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Xác định lớp mục tiêu cho Grad-CAM (lớp tích chập cuối cùng)
        self.target_layer = self.model.layer4[-1] 
        
        # Định nghĩa phần trích xuất đặc trưng (mọi thứ trước lớp FC cuối cùng)
        # Chèn các lớp Dropout cho MCDO. Đối với ResNet, thường tốt sau các khối conv hoặc trước FC.
        # Ở đây, thêm sau avgpool trước FC.
        feature_extractor_layers = list(self.model.children())[:-1]
        
        # Thêm lớp dropout nếu dropout_rate dương và MCDO được bật
        if self.dropout_rate > 0:
            # Tìm chỉ mục của lớp AdaptiveAvgPool2d để chèn dropout sau nó
            # Đây là một vị trí phổ biến để chèn dropout trước lớp tuyến tính cuối cùng
            try:
                avgpool_idx = [i for i, layer in enumerate(feature_extractor_layers) if isinstance(layer, nn.AdaptiveAvgPool2d)][0]
                feature_extractor_layers.insert(avgpool_idx + 1, nn.Dropout(p=self.dropout_rate))
                print(f"Đã thêm lớp Dropout với tỷ lệ {self.dropout_rate} vào bộ trích xuất đặc trưng.")
            except IndexError:
                print("Cảnh báo: Không tìm thấy lớp AdaptiveAvgPool2d. Đang thêm Dropout sau tất cả các lớp tích chập.")
                feature_extractor_layers.append(nn.Dropout(p=self.dropout_rate))


        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        
        # Thay thế lớp kết nối đầy đủ cuối cùng cho phân loại nhị phân
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Forward pass qua bộ trích xuất đặc trưng
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1) # Làm phẳng các đặc trưng
        # Forward pass qua lớp phân loại cuối cùng
        output = self.model.fc(features)
        return output

    def get_features(self, x):
        """
        Trích xuất đặc trưng từ lớp trước đầu phân loại cuối cùng.
        """
        # Lưu ý: Các lớp Dropout trong feature_extractor sẽ tự động tắt nếu model.eval()
        # hoặc hoạt động nếu model.train() và dropout_rate > 0
        with torch.no_grad(): 
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1) 
        return features


# --- 3. Hàm huấn luyện cho các thành viên Ensemble với theo dõi động lực huấn luyện ---
def train_model(model, train_loader, val_loader, epochs, lr, device, model_idx):
    """
    Huấn luyện một thể hiện duy nhất của bộ phân loại cơ bản và theo dõi động lực huấn luyện chi tiết
    (nhãn dự đoán và độ tin cậy cho mỗi mẫu ở mỗi epoch).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.to(device)
    best_val_accuracy = 0.0
    
    # Từ điển để lưu trữ lịch sử dự đoán chi tiết cho mỗi mẫu huấn luyện qua các epoch
    # Key: global_idx của mẫu, Value: Danh sách các dict, mỗi dict (epoch, is_correct, predicted_label, confidence)
    training_prediction_details = {global_idx: [] for global_idx in train_loader.dataset.global_indices}


    print(f"\n--- Huấn luyện Mô hình Ensemble {model_idx + 1} ---")
    for epoch in range(epochs):
        model.train() # Đảm bảo mô hình ở chế độ train (dropout hoạt động nếu có)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels, global_indices_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Tách các outputs trước khi chuyển đổi sang numpy
            probs = softmax(outputs.detach().cpu().numpy(), axis=1) 
            predicted_labels = np.argmax(probs, axis=1)
            confidences_batch = np.max(probs, axis=1)

            total_samples += labels.size(0)
            correct_in_batch = (predicted_labels == labels.cpu().numpy()).sum().item()
            correct_predictions += correct_in_batch

            # Cập nhật động lực huấn luyện: lưu nhãn dự đoán và độ tin cậy của nó cho mỗi mẫu ở epoch này
            for i, global_idx_tensor in enumerate(global_indices_batch):
                global_idx = global_idx_tensor.item()
                is_correct_prediction = (predicted_labels[i] == labels[i].item())
                training_prediction_details[global_idx].append({
                    'epoch': epoch,
                    'is_correct': is_correct_prediction,
                    'predicted_label': predicted_labels[i],
                    'confidence': confidences_batch[i]
                })

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Giai đoạn Validation
        model.eval() # Đặt mô hình về chế độ eval cho validation
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader: # Không cần global_indices trong val_loader cho bước validation
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= val_total_samples

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        # Lưu mô hình tốt nhất dựa trên độ chính xác validation
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{model_idx}.pth'))
            print(f"Đã lưu mô hình tốt nhất {model_idx + 1} với Val Acc: {best_val_accuracy:.4f}")

    print(f"Hoàn thành huấn luyện Mô hình {model_idx + 1}. Val Acc tốt nhất: {best_val_accuracy:.4f}")
    
    # Sau tất cả các epoch cho mô hình này, tính toán 'learning metrics' cho mỗi mẫu
    sample_learning_metrics = {}
    for global_idx, history in training_prediction_details.items():
        correct_epochs_history = [h for h in history if h['is_correct']]
        
        avg_correct_confidence = np.mean([h['confidence'] for h in correct_epochs_history]) if correct_epochs_history else 0.0
        
        # Sự chậm trễ trong học tập: epoch đầu tiên mà nó đúng và vẫn đúng cho tất cả các epoch tiếp theo
        first_correct_epoch = epochs # Mặc định là 'chưa bao giờ học thực sự' (max epochs)
        for i in range(len(history)):
            if history[i]['is_correct']:
                # Kiểm tra xem nó có giữ đúng cho đến cuối không
                if all(h_sub['is_correct'] for h_sub in history[i:]):
                    first_correct_epoch = history[i]['epoch']
                    break
        
        # Tính nhất quán: tỷ lệ dự đoán đúng trên tất cả các epoch cho mô hình này
        consistency = len(correct_epochs_history) / epochs if epochs > 0 else 0.0

        sample_learning_metrics[global_idx] = {
            'avg_correct_confidence': avg_correct_confidence,
            'first_correct_epoch': first_correct_epoch,
            'consistency': consistency
        }
    
    return sample_learning_metrics


# --- 4. Thực thi huấn luyện Ensemble ---
def train_ensemble(cfg, train_loader, val_loader, train_dataset):
    """
    Điều phối việc huấn luyện nhiều bộ phân loại cơ bản cho ensemble.
    Thu thập và tổng hợp động lực huấn luyện chi tiết từ mỗi mô hình.
    """
    # Khởi tạo một từ điển để lưu trữ động lực huấn luyện tổng thể cho mỗi mẫu
    # Giá trị sẽ là danh sách các metrics từ giai đoạn huấn luyện của mỗi mô hình ensemble
    overall_sample_learning_metrics = {global_idx: {'avg_correct_confidence': [], 'first_correct_epoch': [], 'consistency': []}
                                       for global_idx in train_dataset.global_indices}


    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        set_seed(cfg.RANDOM_SEED + i) # Sử dụng các seed khác nhau để đa dạng hóa ensemble
        # Truyền dropout_rate từ cấu hình nếu MCDO được bật
        model = BaseClassifier(num_classes=2, dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0))
        # Huấn luyện mô hình và lấy các learning metrics riêng lẻ của nó
        single_model_learning_metrics = train_model(
            model, train_loader, val_loader, cfg.NUM_EPOCHS_PER_MODEL, cfg.LEARNING_RATE, cfg.DEVICE, i
        )
        # Tổng hợp các learning metrics qua các mô hình ensemble
        for global_idx, metrics in single_model_learning_metrics.items():
            overall_sample_learning_metrics[global_idx]['avg_correct_confidence'].append(metrics['avg_correct_confidence'])
            overall_sample_learning_metrics[global_idx]['first_correct_epoch'].append(metrics['first_correct_epoch'])
            overall_sample_learning_metrics[global_idx]['consistency'].append(metrics['consistency'])

    # Xử lý các metrics tổng hợp để có được 'overall difficulty' cuối cùng cho mỗi mẫu huấn luyện
    final_overall_learning_metrics = {}
    for global_idx, metrics_lists in overall_sample_learning_metrics.items():
        final_overall_learning_metrics[global_idx] = {
            'mean_avg_correct_confidence': np.mean(metrics_lists['avg_correct_confidence']) if metrics_lists['avg_correct_confidence'] else 0.0,
            'mean_first_correct_epoch': np.mean(metrics_lists['first_correct_epoch']) if metrics_lists['first_correct_epoch'] else cfg.NUM_EPOCHS_PER_MODEL, # Nếu không bao giờ học được, sử dụng epoch tối đa
            'mean_consistency': np.mean(metrics_lists['consistency']) if metrics_lists['consistency'] else 0.0
        }
    
    print("\n--- Huấn luyện Ensemble Hoàn tất ---")
    return final_overall_learning_metrics


# --- 5. Ước tính và hiệu chỉnh độ tin cậy nâng cao ---
class TemperatureScaler(nn.Module):
    """
    Học một tham số nhiệt độ scalar duy nhất để hiệu chỉnh các xác suất.
    Dựa trên Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits_to_calibrate, labels_for_calibration, device):
        """
        Điều chỉnh tham số nhiệt độ bằng cách sử dụng các logits và nhãn đã tính toán trước.
        """
        # Đảm bảo logits và nhãn nằm trên thiết bị chính xác
        logits_all = logits_to_calibrate.to(device)
        labels_all = labels_for_calibration.to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Bộ hiệu chỉnh nhiệt độ đã được hiệu chỉnh. Tối ưu T: {self.temperature.item():.4f}")


def extract_features(model, data_loader, device):
    """
    Trích xuất đặc trưng từ feature_extractor của mô hình cho tất cả các mẫu trong data_loader.
    Trả về các đặc trưng dưới dạng mảng numpy và các chỉ mục toàn cục gốc tương ứng.
    """
    model.eval() # Đảm bảo model ở chế độ eval cho việc trích xuất đặc trưng thông thường
    all_features = []
    all_indices = []
    with torch.no_grad():
        for inputs, _, global_indices_batch in tqdm(data_loader, desc="Trích xuất Đặc trưng"):
            inputs = inputs.to(device)
            features = model.get_features(inputs) # Sử dụng phương thức get_features mới
            all_features.append(features.cpu().numpy())
            all_indices.extend(global_indices_batch.cpu().numpy())
    return np.vstack(all_features), np.array(all_indices)


def adjust_confidence_with_training_dynamics(cfg, test_features, test_confidences,
                                             train_features, train_global_indices, final_overall_learning_metrics):
    """
    Điều chỉnh độ tin cậy của tập test dựa trên sự tương đồng với các mẫu huấn luyện và động lực học của chúng.
    Độ tin cậy thấp hơn đối với các mẫu test tương tự các mẫu huấn luyện 'khó' (ví dụ: học muộn, không nhất quán).
    """
    print("Điều chỉnh độ tin cậy của tập test với động lực huấn luyện...")
    adjusted_confidences = np.copy(test_confidences)

    # Tạo ánh xạ từ global_idx đến từ điển learning metrics để tra cứu hiệu quả
    train_global_idx_to_metrics = {idx: metrics for idx, metrics in final_overall_learning_metrics.items()}

    # Tính toán độ tương đồng cosine giữa các đặc trưng test và huấn luyện
    if len(test_features) == 0 or len(train_features) == 0:
        print("Bỏ qua điều chỉnh động lực huấn luyện: Không có đặc trưng test hoặc huấn luyện.")
        return adjusted_confidences

    similarities = cosine_similarity(test_features, train_features)
    
    for i in tqdm(range(len(test_features)), desc="Áp dụng điều chỉnh động lực huấn luyện"):
        # Tìm mẫu huấn luyện tương đồng nhất (bằng chỉ mục của nó trong mảng `train_features`)
        most_similar_train_idx_in_features_array = np.argmax(similarities[i])
        # Lấy chỉ mục toàn cục gốc của mẫu huấn luyện tương đồng nhất đó
        most_similar_train_global_idx = train_global_indices[most_similar_train_idx_in_features_array]
        
        # Kiểm tra xem các learning metrics cho chỉ mục toàn cục này có tồn tại không
        if most_similar_train_global_idx in train_global_idx_to_metrics:
            sample_metrics = train_global_idx_to_metrics[most_similar_train_global_idx]
            
            # Sử dụng 'mean_first_correct_epoch' làm proxy cho 'learning lateness' hoặc 'difficulty'.
            # 'mean_first_epoch' cao hơn cho thấy một mẫu khó học hơn.
            # Sử dụng .get() với giá trị mặc định trong trường hợp key bị thiếu bất ngờ
            difficulty_value = sample_metrics.get('mean_first_correct_epoch', cfg.NUM_EPOCHS_PER_MODEL)
            
            # Chuẩn hóa độ khó nằm giữa 0 và 1 (0 = dễ, 1 = khó).
            # Nếu một mẫu được học muộn (epoch cao hơn), nó khó hơn, vì vậy độ khó chuẩn hóa gần 1.
            if cfg.NUM_EPOCHS_PER_MODEL > 0:
                normalized_difficulty = difficulty_value / cfg.NUM_EPOCHS_PER_MODEL
            else:
                normalized_difficulty = 0.0 # Mặc định nếu không có epoch nào được định nghĩa (hoặc 0.5 cho trung tính)


        else:
            # Nếu không tìm thấy chỉ mục (ví dụ: một mẫu huấn luyện bằng cách nào đó bị bỏ lỡ trong quá trình theo dõi),
            # mặc định không có hình phạt (độ khó trung tính).
            normalized_difficulty = 0.0 


        # Giảm độ tin cậy dựa trên độ khó và một yếu tố hình phạt có thể điều chỉnh (cfg.TRAINING_DYNAMICS_CONF_PENALTY)
        # Độ khó cao hơn dẫn đến giảm độ tin cậy lớn hơn
        adjustment_factor = 1.0 - (cfg.TRAINING_DYNAMICS_CONF_PENALTY * normalized_difficulty)
        
        adjusted_confidences[i] *= adjustment_factor
        adjusted_confidences[i] = max(0.0, adjusted_confidences[i]) # Đảm bảo độ tin cậy không âm

    return adjusted_confidences


def get_confidence_and_predictions(cfg, data_loader, ensemble_models_dir, final_overall_learning_metrics=None, train_dataset=None, train_features=None, train_indices=None):
    """
    Tính toán điểm độ tin cậy và dự đoán của ensemble cho các mẫu, tích hợp Monte Carlo Dropout (MCDO).
    Áp dụng hiệu chỉnh nhiệt độ và kết hợp sự bất đồng của ensemble (bao gồm MCDO).
    Tùy chọn áp dụng điều chỉnh dựa trên động lực huấn luyện nếu `final_overall_learning_metrics` và `train_dataset` được cung cấp.

    Trả về:
    - all_predictions: Các dự đoán ensemble cuối cùng
    - all_confidences: Điểm độ tin cậy đã điều chỉnh cuối cùng
    - all_labels: Nhãn thực
    - all_original_indices: Các chỉ mục toàn cục gốc của các mẫu
    - all_ensemble_individual_predictions: Danh sách các danh sách, mỗi danh sách con chứa các dự đoán cho một mô hình ensemble (đã làm phẳng).
    - all_ensemble_individual_probs_stacked: Mảng NumPy (num_samples, total_runs_per_sample, num_classes) của các xác suất mô hình riêng lẻ từ tất cả các chạy (ensemble + MCDO).
    """
    all_predictions = []
    all_confidences = []
    all_labels = []
    all_original_indices = []
    
    # Lưu trữ các dự đoán mô hình ensemble riêng lẻ (danh sách phẳng cho mỗi mô hình)
    # Lưu ý: Nếu MCDO bật, đây chỉ là dự đoán trung bình của MCDO cho mỗi mô hình ensemble
    all_ensemble_individual_predictions = [[] for _ in range(cfg.NUM_ENSEMBLE_MODELS)]
    
    # Lưu trữ TẤT CẢ các xác suất từ MỌI chạy (ensemble models x MCDO runs) cho mỗi mẫu
    # Cấu trúc: list of lists, where each inner list contains batch_size x (num_classes) arrays
    raw_probs_from_all_runs_per_batch = [] 


    # Tải tất cả các mô hình ensemble
    loaded_models = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        # Đảm bảo dropout_rate được truyền trong constructor để kích hoạt MCDO
        model = BaseClassifier(num_classes=2, dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(ensemble_models_dir, f'best_model_ensemble_{i}.pth')))
        # Đặt lại chế độ eval sau khi load.
        # Đối với MCDO, chúng ta sẽ đặt lại thành train() trong vòng lặp dự đoán.
        model.eval() 
        loaded_models.append(model)

    temp_scaler = TemperatureScaler()
    temp_scaler.to(cfg.DEVICE)

    print("Hiệu chỉnh bộ hiệu chỉnh nhiệt độ trên các logits trung bình của ensemble từ trình tải dữ liệu hiện tại...")
    # --- Thu thập tất cả các logits trung bình của ensemble từ data_loader hiện tại để hiệu chỉnh ---
    data_loader_ensemble_avg_logits = []
    data_loader_labels_for_calibration = []
    with torch.no_grad():
        for inputs_batch_cal, labels_batch_cal, _ in tqdm(data_loader, desc="Thu thập Logits để hiệu chỉnh"):
            inputs_batch_cal = inputs_batch_cal.to(cfg.DEVICE)
            
            ensemble_logits_batch_cal_runs = [] # Logits cho batch hiện tại từ tất cả các chạy ensemble/MCDO
            
            for model in loaded_models:
                if cfg.MCDO_ENABLE:
                    model.train() # Kích hoạt dropout cho MCDO
                    mcd_logits_for_current_model = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        mcd_logits_for_current_model.append(model(inputs_batch_cal))
                    ensemble_logits_batch_cal_runs.append(torch.mean(torch.stack(mcd_logits_for_current_model), dim=0))
                    model.eval() # Tắt dropout sau các chạy MCDO cho mô hình này
                else:
                    ensemble_logits_batch_cal_runs.append(model(inputs_batch_cal))

            # This is still averaging all models for calibration. This is correct as calibration happens
            # on the overall ensemble output.
            avg_logits_batch_cal = torch.stack(ensemble_logits_batch_cal_runs).mean(dim=0) # Trung bình hóa ensemble
            
            data_loader_ensemble_avg_logits.append(avg_logits_batch_cal.cpu())
            data_loader_labels_for_calibration.append(labels_batch_cal.cpu())
    
    if data_loader_ensemble_avg_logits: # Kiểm tra xem danh sách có trống không
        data_loader_ensemble_avg_logits_all = torch.cat(data_loader_ensemble_avg_logits).to(cfg.DEVICE)
        data_loader_labels_for_calibration_all = torch.cat(data_loader_labels_for_calibration).to(cfg.DEVICE)
        # Hiệu chỉnh bộ hiệu chỉnh nhiệt độ bằng cách sử dụng các logits trung bình của ensemble
        temp_scaler.calibrate(data_loader_ensemble_avg_logits_all, data_loader_labels_for_calibration_all, cfg.DEVICE)
    else:
        print("Cảnh báo: Không có dữ liệu trong data_loader để hiệu chỉnh. Bỏ qua hiệu chỉnh nhiệt độ.")


    # --- Trích xuất đặc trưng từ data_loader hiện tại (tập val/test) *một lần* ---
    # `all_extracted_features` cần được định nghĩa ở đây cho phạm vi của hàm này.
    print("Trích xuất đặc trưng từ trình tải dữ liệu hiện tại để điều chỉnh độ tin cậy...")
    # Sử dụng mô hình đầu tiên để trích xuất đặc trưng, đặt nó về eval() để dropout không ảnh hưởng đến đặc trưng
    loaded_models[0].eval() 
    all_extracted_features, extracted_original_indices = extract_features(
        loaded_models[0], data_loader, cfg.DEVICE 
    )

    print("Tạo dự đoán và độ tin cậy...")
    with torch.no_grad(): # Tắt grad tổng thể, nhưng bật lại cho MCDO models.train()
        for inputs, labels, original_indices_batch in tqdm(data_loader, desc="Dự đoán"):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            # This will store logits for each model for the current batch
            # Shape: (num_ensemble_models, batch_size, num_classes)
            individual_model_logits_for_batch = [] 
            
            # This will store probabilities from all individual runs (models * MCDO runs)
            current_batch_all_probs_runs = []

            for model_idx, model in enumerate(loaded_models):
                if cfg.MCDO_ENABLE:
                    model.train() # Kích hoạt dropout cho MCDO
                    mcd_logits_for_current_model = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        logits_run = model(inputs)
                        mcd_logits_for_current_model.append(logits_run)
                        # Store raw probabilities for disagreement calculation later
                        current_batch_all_probs_runs.append(softmax(logits_run.detach().cpu().numpy(), axis=1)) 
                    # Average MCDO logits for this specific model
                    avg_mcd_logits = torch.mean(torch.stack(mcd_logits_for_current_model), dim=0)
                    individual_model_logits_for_batch.append(avg_mcd_logits)
                    # For individual prediction tracking, use the averaged MCDO probs
                    avg_mcd_probs = softmax(avg_mcd_logits.detach().cpu().numpy(), axis=1)
                    all_ensemble_individual_predictions[model_idx].extend(np.argmax(avg_mcd_probs, axis=1))
                    model.eval() # Tắt dropout sau các chạy MCDO cho mô hình này
                else:
                    logits = model(inputs)
                    individual_model_logits_for_batch.append(logits)
                    current_batch_all_probs_runs.append(softmax(logits.detach().cpu().numpy(), axis=1))
                    all_ensemble_individual_predictions[model_idx].extend(np.argmax(softmax(logits.detach().cpu().numpy(), axis=1), axis=1))

            # Stack individual model logits to (batch_size, num_ensemble_models, num_classes)
            individual_model_logits_for_batch_stacked = torch.stack(individual_model_logits_for_batch, dim=1)
            
            # Prepare a tensor to store the gated ensemble logits for the current batch
            gated_ensemble_logits_for_batch = torch.zeros_like(individual_model_logits_for_batch_stacked[:, 0, :])

            # Apply dynamic selection per sample in the batch
            if cfg.USE_DYNAMIC_SELECTION and cfg.NUM_ENSEMBLE_MODELS > 1 and cfg.DYNAMIC_SELECTION_COUNT < cfg.NUM_ENSEMBLE_MODELS:
                for sample_idx in range(inputs.size(0)):
                    # Get logits for all models for the current sample
                    sample_individual_logits = individual_model_logits_for_batch_stacked[sample_idx, :, :] # (NUM_ENSEMBLE_MODELS, num_classes)
                    
                    # Convert logits to probabilities to determine confidence for selection
                    sample_individual_probs = softmax(sample_individual_logits.detach().cpu().numpy(), axis=1)
                    
                    # Get the confidence for the predicted class for each model
                    # This assumes we want to select models that are most confident in their *own* prediction
                    model_confidences = np.max(sample_individual_probs, axis=1)
                    
                    # Sort models by confidence in descending order and get their original indices
                    sorted_model_indices = np.argsort(model_confidences)[::-1]
                    
                    # Select the top DYNAMIC_SELECTION_COUNT models
                    selected_model_indices = sorted_model_indices[:cfg.DYNAMIC_SELECTION_COUNT]
                    
                    # Gather logits from selected models for this sample
                    selected_logits = sample_individual_logits[selected_model_indices, :]
                    
                    # Average the logits from the selected models
                    gated_ensemble_logits_for_batch[sample_idx, :] = torch.mean(selected_logits, dim=0)
            else:
                # If dynamic selection is off or not applicable, average all models (original behavior)
                gated_ensemble_logits_for_batch = individual_model_logits_for_batch_stacked.mean(dim=1)

            # Nối các xác suất từ tất cả các chạy vào danh sách tổng thể
            if current_batch_all_probs_runs:
                # Each element in current_batch_all_probs_runs is a (batch_size, num_classes) array
                num_total_runs_per_sample = cfg.NUM_ENSEMBLE_MODELS * (cfg.MCDO_NUM_RUNS if cfg.MCDO_ENABLE else 1)
                
                # Stack them: result is (num_total_runs_per_sample, batch_size, num_classes)
                stacked_for_disagreement_temp = np.stack(current_batch_all_probs_runs, axis=0)
                # Transpose to (batch_size, num_total_runs_per_sample, num_classes)
                stacked_for_disagreement = np.transpose(stacked_for_disagreement_temp, (1, 0, 2))
                raw_probs_from_all_runs_per_batch.append(stacked_for_disagreement)
            
            # Use the gated_ensemble_logits_for_batch for temperature scaling
            calibrated_logits_ensemble = temp_scaler.forward(gated_ensemble_logits_for_batch) 
            
            # Chuyển đổi logits đã hiệu chỉnh thành xác suất và dự đoán
            calibrated_probs_ensemble = softmax(calibrated_logits_ensemble.detach().cpu().numpy(), axis=1)
            predicted_labels_ensemble = np.argmax(calibrated_probs_ensemble, axis=1)
            confidences_ensemble = np.max(calibrated_probs_ensemble, axis=1)

            all_predictions.extend(predicted_labels_ensemble)
            all_confidences.extend(confidences_ensemble)
            all_labels.extend(labels.cpu().numpy())
            all_original_indices.extend(original_indices_batch.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_labels = np.array(all_labels)
    all_original_indices = np.array(all_original_indices)
    
    # Nối tất cả các mảng xác suất thô từ các chạy lại với nhau
    all_ensemble_individual_probs_stacked = np.concatenate(raw_probs_from_all_runs_per_batch, axis=0)

    # Áp dụng điều chỉnh động lực huấn luyện nếu được bật và dữ liệu được cung cấp
    if cfg.ENABLE_TRAINING_DYNAMICS and final_overall_learning_metrics is not None:
        # train_features và train_indices được truyền trực tiếp vào hàm này
        if train_features is not None and train_indices is not None:
            all_confidences = adjust_confidence_with_training_dynamics(
                cfg, all_extracted_features, all_confidences,
                train_features, train_indices, final_overall_learning_metrics
            )
        else:
            print("Cảnh báo: Không có đặc trưng huấn luyện hoặc chỉ mục được cung cấp cho điều chỉnh động lực huấn luyện.")


    # Tính toán sự bất đồng của ensemble (Ensemble Disagreement)
    # Đây là độ lệch chuẩn của các xác suất dự đoán trên tất cả các mô hình ensemble và các chạy MCDO.
    # Một độ lệch chuẩn cao hơn cho thấy sự bất đồng lớn hơn.
    # all_ensemble_individual_probs_stacked: (num_samples, total_runs_per_sample, num_classes)
    # Lấy xác suất của lớp được dự đoán bởi ensemble cuối cùng
    # Lấy index của lớp được dự đoán bởi ensemble cuối cùng
    predicted_classes_indices = all_predictions
    
    # Tạo một mảng để lưu trữ xác suất của lớp được dự đoán bởi ensemble cho mỗi chạy riêng lẻ
    probs_of_predicted_class_per_run = np.zeros((all_ensemble_individual_probs_stacked.shape[0], all_ensemble_individual_probs_stacked.shape[1]))
    
    for i in range(all_ensemble_individual_probs_stacked.shape[0]): # Duyệt qua từng mẫu
        for j in range(all_ensemble_individual_probs_stacked.shape[1]): # Duyệt qua từng chạy (mô hình + MCDO)
            # Lấy xác suất của lớp được dự đoán bởi ensemble cuối cùng cho chạy này
            probs_of_predicted_class_per_run[i, j] = all_ensemble_individual_probs_stacked[i, j, predicted_classes_indices[i]]

    # Tính độ lệch chuẩn của xác suất của lớp được dự đoán trên các chạy
    disagreement_scores = np.std(probs_of_predicted_class_per_run, axis=1)

    # Áp dụng hình phạt bất đồng: Độ bất đồng cao hơn -> độ tin cậy thấp hơn
    # Sử dụng một hàm giảm dần, ví dụ: 1 / (1 + factor * disagreement)
    disagreement_penalty = 1.0 / (1.0 + cfg.DISAGREEMENT_PENALTY_FACTOR * disagreement_scores)
    all_confidences *= disagreement_penalty
    all_confidences = np.clip(all_confidences, 0.0, 1.0) # Đảm bảo độ tin cậy nằm trong [0, 1]

    return all_predictions, all_confidences, all_labels, all_original_indices, all_ensemble_individual_predictions, all_ensemble_individual_probs_stacked


# --- 6. Các hàm đánh giá hiệu suất ---
def calculate_ece(confidences, predictions, true_labels, num_bins=10):
    """
    Tính toán Expected Calibration Error (ECE).
    Chia các dự đoán thành các bin dựa trên độ tin cậy, sau đó tính toán
    độ chính xác và độ tin cậy trung bình cho mỗi bin.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        lower_bound = bin_boundaries[i]
        upper_bound = bin_boundaries[i+1]
        
        # Tìm các mẫu trong bin hiện tại
        in_bin = (confidences > lower_bound) & (confidences <= upper_bound)
        
        if np.any(in_bin):
            bin_confidences = confidences[in_bin]
            bin_predictions = predictions[in_bin]
            bin_true_labels = true_labels[in_bin]
            
            # Tính độ chính xác và độ tin cậy trung bình cho bin
            accuracy_in_bin = np.mean(bin_predictions == bin_true_labels)
            avg_confidence_in_bin = np.mean(bin_confidences)
            
            # Trọng số của bin (tỷ lệ mẫu trong bin)
            bin_weight = len(bin_confidences) / total_samples
            
            ece += bin_weight * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return ece


def plot_calibration_curve(predictions, confidences, true_labels, num_bins=10, save_path=None):
    """
    Vẽ biểu đồ đường cong hiệu chỉnh (reliability diagram).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    accuracies = []
    avg_confidences = []

    for i in range(num_bins):
        lower_bound = bin_boundaries[i]
        upper_bound = bin_boundaries[i+1]
        
        in_bin = (confidences > lower_bound) & (confidences <= upper_bound)
        
        if np.any(in_bin):
            bin_predictions = predictions[in_bin]
            bin_true_labels = true_labels[in_bin]
            bin_confidences = confidences[in_bin]

            accuracies.append(np.mean(bin_predictions == bin_true_labels))
            avg_confidences.append(np.mean(bin_confidences))
        else:
            accuracies.append(np.nan) # Sử dụng NaN để không vẽ điểm nếu bin trống
            avg_confidences.append(np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(avg_confidences, accuracies, marker='o', linestyle='-', color='blue', label='Model')
    plt.xlabel('Average Confidence (Bin)')
    plt.ylabel('Accuracy (Bin)')
    plt.title('Reliability Diagram')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(predictions, confidences, true_labels, save_path=None):
    """
    Vẽ biểu đồ đường cong ROC và tính toán AUC.
    """
    fpr, tpr, _ = roc_curve(true_labels, confidences)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_precision_recall_curve(predictions, confidences, true_labels, save_path=None):
    """
    Vẽ biểu đồ đường cong Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(true_labels, confidences)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', alpha=0.7, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(predictions, true_labels, class_names=['Non-Abnormal', 'Abnormal'], save_path=None):
    """
    Vẽ ma trận nhầm lẫn.
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# --- 7. Phân loại có chọn lọc và XAI ---
def optimize_rejection_threshold(cfg, confidences, true_labels):
    """
    Tìm ngưỡng từ chối tối ưu dựa trên mục tiêu độ chính xác và tỷ lệ từ chối.
    Sử dụng tìm kiếm lưới để đánh giá các ngưỡng khác nhau và chọn ngưỡng tốt nhất
    dựa trên hàm mục tiêu kết hợp.
    """
    best_threshold = 0.0
    min_objective_value = float('inf')
    
    # Các ngưỡng tiềm năng để thử
    threshold_candidates = np.linspace(0.0, 1.0, 100) # 100 điểm từ 0 đến 1

    results = []

    for threshold in tqdm(threshold_candidates, desc="Tối ưu hóa ngưỡng từ chối"):
        accepted_mask = confidences >= threshold
        rejected_mask = confidences < threshold

        num_accepted = np.sum(accepted_mask)
        num_rejected = np.sum(rejected_mask)
        
        rejection_rate = num_rejected / len(confidences) if len(confidences) > 0 else 0.0

        accepted_accuracy = 0.0
        ece_accepted = 0.0

        if num_accepted > 0:
            accepted_confidences = confidences[accepted_mask]
            accepted_predictions = (confidences[accepted_mask] >= 0.5).astype(int) # Dự đoán dựa trên độ tin cậy
            accepted_true_labels = true_labels[accepted_mask]
            
            accepted_accuracy = accuracy_score(accepted_true_labels, accepted_predictions)
            ece_accepted = calculate_ece(accepted_confidences, accepted_predictions, accepted_true_labels)
        
        # Tính toán độ lệch so với mục tiêu
        accuracy_deviation = np.abs(accepted_accuracy - cfg.TARGET_ACCEPTED_ACCURACY)
        rejection_rate_deviation = np.abs(rejection_rate - cfg.TARGET_REJECTION_RATE)
        
        # Hàm mục tiêu: Giảm thiểu độ lệch, với trọng số cho mỗi thành phần
        # ECE được thêm vào để khuyến khích hiệu chỉnh tốt trong tập chấp nhận
        objective_value = (cfg.ACCURACY_DEVIATION_WEIGHT * accuracy_deviation +
                          cfg.REJECTION_RATE_DEVIATION_WEIGHT * rejection_rate_deviation +
                          cfg.ECE_DEVIATION_WEIGHT * ece_accepted)
        
        results.append({
            'threshold': threshold,
            'accepted_accuracy': accepted_accuracy,
            'rejection_rate': rejection_rate,
            'ece_accepted': ece_accepted,
            'objective_value': objective_value
        })

        if objective_value < min_objective_value:
            min_objective_value = objective_value
            best_threshold = threshold
            
    print(f"Ngưỡng từ chối tối ưu: {best_threshold:.4f}")
    best_result = [r for r in results if r['threshold'] == best_threshold][0]
    print(f"  Độ chính xác được chấp nhận: {best_result['accepted_accuracy']:.4f}")
    print(f"  Tỷ lệ từ chối: {best_result['rejection_rate']:.4f}")
    print(f"  ECE trên tập được chấp nhận: {best_result['ece_accepted']:.4f}")
    print(f"  Giá trị mục tiêu: {best_result['objective_value']:.4f}")

    return best_threshold, results


def visualize_xai_examples(cfg, models, dataset, test_results, rejection_threshold, num_examples=5):
    """
    Trực quan hóa các ví dụ được chấp nhận và từ chối cùng với Grad-CAM.
    """
    print("\n--- Trực quan hóa các ví dụ XAI ---")
    
    model_preds = test_results['model_preds']
    confidences = test_results['confidences']
    true_labels = test_results['true_labels']
    original_indices = test_results['original_indices']
    rejected_categories = test_results['rejected_categories']

    # Lấy các chỉ mục của các mẫu được chấp nhận/từ chối
    accepted_mask = confidences >= rejection_threshold
    rejected_mask = confidences < rejection_threshold

    accepted_indices_in_test_results = np.where(accepted_mask)[0]
    rejected_indices_in_test_results = np.where(rejected_mask)[0]

    # Chọn ngẫu nhiên các ví dụ từ mỗi loại
    selected_accepted_indices = np.random.choice(accepted_indices_in_test_results, min(num_examples, len(accepted_indices_in_test_results)), replace=False)
    selected_rejected_indices = np.random.choice(rejected_indices_in_test_results, min(num_examples, len(rejected_indices_in_test_results)), replace=False)

    # Đảm bảo có ít nhất một mô hình để sử dụng Grad-CAM
    if not models:
        print("Không có mô hình nào được cung cấp để tạo Grad-CAM. Bỏ qua trực quan hóa Grad-CAM.")
        return

    # Sử dụng mô hình đầu tiên trong ensemble cho Grad-CAM
    # Đảm bảo mô hình ở chế độ eval() cho Grad-CAM
    grad_cam_model = models[0]
    grad_cam_model.eval() 
    target_layer = grad_cam_model.target_layer
    # Đã sửa lỗi: Loại bỏ tham số use_cuda
    cam = GradCAMPlusPlus(model=grad_cam_model, target_layers=[target_layer])

    # Hàm trợ giúp để hiển thị hình ảnh và Grad-CAM
    def show_image_with_cam(ax, image_tensor, heatmap, title):
        # Chuyển đổi tensor hình ảnh trở lại định dạng hiển thị
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        # Chuẩn hóa lại hình ảnh nếu nó được chuẩn hóa
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        # Tạo hình ảnh với heatmap
        cam_image = show_cam_on_image(img_np, heatmap, use_rgb=True)
        ax.imshow(cam_image)
        ax.set_title(title)
        ax.axis('off')

    # Trực quan hóa các ví dụ được chấp nhận
    print("\n--- Ví dụ được chấp nhận (Accept) ---")
    plt.figure(figsize=(num_examples * 3, 4))
    for i, test_idx in enumerate(selected_accepted_indices):
        original_global_idx = original_indices[test_idx]
        # Tìm chỉ mục cục bộ trong test_dataset
        local_idx_in_test_dataset = dataset.original_indices_map[original_global_idx]
        
        image_tensor, true_label, _ = dataset[local_idx_in_test_dataset]
        
        # Thêm chiều batch
        input_tensor = image_tensor.unsqueeze(0).to(cfg.DEVICE)
        
        predicted_label = model_preds[test_idx]
        confidence = confidences[test_idx]
        
        # Tạo heatmap Grad-CAM
        targets = [ClassifierOutputTarget(predicted_label)] # Mục tiêu là lớp dự đoán
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Lấy heatmap đầu tiên

        ax = plt.subplot(1, num_examples, i + 1)
        title = (f"True: {true_label}\nPred: {predicted_label}\n"
                 f"Conf: {confidence:.2f}")
        show_image_with_cam(ax, image_tensor, grayscale_cam, title)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.XAI_SAVE_DIR, 'accepted_examples_cam.png'))
    plt.show()

    # Trực quan hóa các ví dụ bị từ chối
    print("\n--- Ví dụ bị từ chối (Reject) ---")
    plt.figure(figsize=(num_examples * 3, 4))
    for i, test_idx in enumerate(selected_rejected_indices):
        original_global_idx = original_indices[test_idx]
        # Tìm chỉ mục cục bộ trong test_dataset
        local_idx_in_test_dataset = dataset.original_indices_map[original_global_idx]

        image_tensor, true_label, _ = dataset[local_idx_in_test_dataset]
        
        # Thêm chiều batch
        input_tensor = image_tensor.unsqueeze(0).to(cfg.DEVICE)

        predicted_label = model_preds[test_idx]
        confidence = confidences[test_idx]
        rejection_reason = "Unknown"
        # Tìm lý do từ chối cụ thể nếu có
        for reason, indices in rejected_categories.items():
            if test_idx in indices:
                rejection_reason = reason
                break

        # Tạo heatmap Grad-CAM
        targets = [ClassifierOutputTarget(predicted_label)] # Mục tiêu là lớp dự đoán
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Lấy heatmap đầu tiên

        ax = plt.subplot(1, num_examples, i + 1)
        title = (f"True: {true_label}\nPred: {predicted_label}\n"
                 f"Conf: {confidence:.2f}\nRejected: {rejection_reason}")
        show_image_with_cam(ax, image_tensor, grayscale_cam, title)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.XAI_SAVE_DIR, 'rejected_examples_cam.png'))
    plt.show()


# --- Hàm chính để chạy toàn bộ quy trình ---
def run_selective_classification_pipeline(cfg):
    """
    Chạy toàn bộ quy trình phân loại có chọn lọc:
    1. Chuẩn bị dữ liệu.
    2. Huấn luyện ensemble các mô hình.
    3. Ước tính độ tin cậy và dự đoán của ensemble với hiệu chỉnh nhiệt độ và MCDO.
    4. Tối ưu hóa ngưỡng từ chối.
    5. Đánh giá hiệu suất phân loại có chọn lọc.
    6. Trực quan hóa các ví dụ XAI.
    """
    print("--- Bắt đầu quy trình phân loại có chọn lọc ---")

    # 1. Chuẩn bị dữ liệu
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_datasets(cfg)

    # 2. Huấn luyện ensemble các mô hình
    # Động lực huấn luyện được thu thập trong quá trình này
    final_overall_learning_metrics = train_ensemble(cfg, train_loader, val_loader, train_dataset)

    # 3. Ước tính độ tin cậy và dự đoán của ensemble
    # Sử dụng test_loader để có được dự đoán trên tập test độc lập
    print("\n--- Đánh giá hiệu suất Ensemble trên tập Test ---")
    
    # Tạo một DataLoader cho tập huấn luyện để trích xuất đặc trưng
    train_features_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    # Tải lại các mô hình để đảm bảo chúng ở đúng chế độ (eval) và có thể được sử dụng cho XAI và trích xuất đặc trưng
    loaded_models_for_feature_extraction = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(num_classes=2, dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{i}.pth')))
        model.eval() 
        loaded_models_for_feature_extraction.append(model)

    # Trích xuất đặc trưng huấn luyện một lần ở đây
    # Sử dụng mô hình đầu tiên để trích xuất đặc trưng
    train_features_for_adjustment, train_indices_for_adjustment = extract_features(
        loaded_models_for_feature_extraction[0], train_features_loader, cfg.DEVICE
    )

    # Truyền train_features_for_adjustment và train_indices_for_adjustment vào get_confidence_and_predictions
    test_model_predictions, test_confidences, test_true_labels, test_original_indices, \
    test_ensemble_individual_predictions, test_ensemble_individual_probs_stacked = \
        get_confidence_and_predictions(cfg, test_loader, cfg.MODEL_SAVE_DIR, 
                                       final_overall_learning_metrics, 
                                       train_dataset, # Truyền train_dataset để có thể truy cập global_indices
                                       train_features=train_features_for_adjustment, 
                                       train_indices=train_indices_for_adjustment)


    # 4. Tối ưu hóa ngưỡng từ chối
    print("\n--- Tối ưu hóa ngưỡng từ chối ---")
    best_rejection_threshold, rejection_optimization_results = optimize_rejection_threshold(
        cfg, test_confidences, test_true_labels
    )

    # 5. Đánh giá hiệu suất phân loại có chọn lọc
    print("\n--- Đánh giá hiệu suất phân loại có chọn lọc ---")
    accepted_mask = test_confidences >= best_rejection_threshold
    rejected_mask = test_confidences < best_rejection_threshold

    accepted_predictions = test_model_predictions[accepted_mask]
    accepted_confidences = test_confidences[accepted_mask]
    accepted_true_labels = test_true_labels[accepted_mask]

    rejected_predictions = test_model_predictions[rejected_mask]
    rejected_confidences = test_confidences[rejected_mask]
    rejected_true_labels = test_true_labels[rejected_mask]
    
    num_total_samples = len(test_confidences)
    num_accepted_samples = len(accepted_confidences)
    num_rejected_samples = len(rejected_confidences)

    print(f"Tổng số mẫu: {num_total_samples}")
    print(f"Số mẫu được chấp nhận: {num_accepted_samples} ({num_accepted_samples/num_total_samples:.2%})")
    print(f"Số mẫu bị từ chối: {num_rejected_samples} ({num_rejected_samples/num_total_samples:.2%})")

    if num_accepted_samples > 0:
        accepted_accuracy = accuracy_score(accepted_true_labels, accepted_predictions)
        accepted_ece = calculate_ece(accepted_confidences, accepted_predictions, accepted_true_labels)
        print(f"Độ chính xác trên các mẫu được chấp nhận: {accepted_accuracy:.4f}")
        print(f"ECE trên các mẫu được chấp nhận: {accepted_ece:.4f}")
        plot_confusion_matrix(accepted_predictions, accepted_true_labels, save_path=os.path.join(cfg.XAI_SAVE_DIR, 'confusion_matrix_accepted.png'))
    else:
        print("Không có mẫu nào được chấp nhận.")

    if num_rejected_samples > 0:
        rejected_accuracy = accuracy_score(rejected_true_labels, rejected_predictions)
        print(f"Độ chính xác trên các mẫu bị từ chối (chỉ để tham khảo): {rejected_accuracy:.4f}")
        
        # Phân loại các mẫu bị từ chối theo lý do (độ tin cậy thấp, bất đồng cao, OOD)
        rejected_categories_info = {
            'low_confidence': np.where((test_confidences < best_rejection_threshold) & (test_confidences < cfg.OOD_CONFIDENCE_THRESHOLD))[0],
            'high_disagreement': np.where((test_confidences < best_rejection_threshold) & (test_confidences >= cfg.OOD_CONFIDENCE_THRESHOLD))[0], # Placeholder for now, need actual disagreement metric
            # OOD (Out-of-Distribution) detection heuristic
            'ood_potential': np.where((test_confidences < cfg.OOD_CONFIDENCE_THRESHOLD) & 
                                      (np.std(test_ensemble_individual_probs_stacked[:, :, 1], axis=1) > cfg.OOD_VARIANCE_THRESHOLD))[0] # Variance of abnormal class probability
        }
        test_results_for_xai = {
            'model_preds': test_model_predictions,
            'confidences': test_confidences,
            'true_labels': test_true_labels,
            'original_indices': test_original_indices,
            'rejected_categories': rejected_categories_info
        }
    else:
        print("Không có mẫu nào bị từ chối.")
        rejected_categories_info = {}
        test_results_for_xai = {
            'model_preds': test_model_predictions,
            'confidences': test_confidences,
            'true_labels': test_true_labels,
            'original_indices': test_original_indices,
            'rejected_categories': {}
        }
    
    # 6. Trực quan hóa các ví dụ XAI
    # Tải lại các mô hình để đảm bảo chúng ở đúng chế độ (eval) và có thể được sử dụng cho XAI
    loaded_ensemble_models_for_xai = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(num_classes=2, dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{i}.pth')))
        model.eval() # Đảm bảo mô hình ở chế độ eval() cho Grad-CAM một khi dự đoán đã được xác định.
        loaded_ensemble_models_for_xai.append(model)
        
    test_results_for_xai = {
        'model_preds': test_model_predictions,
        'confidences': test_confidences,
        'true_labels': test_true_labels,
        'original_indices': test_original_indices,
        'rejected_categories': rejected_categories_info
    }
    
    print(f"Đảm bảo thư mục lưu XAI tồn tại: {cfg.XAI_SAVE_DIR}")
    os.makedirs(cfg.XAI_SAVE_DIR, exist_ok=True)
    visualize_xai_examples(cfg, loaded_ensemble_models_for_xai, test_dataset, test_results_for_xai, best_rejection_threshold)

    # 8. Trực quan hóa các đường cong hiệu suất
    print("\n--- Trực quan hóa các đường cong hiệu suất ---")
    plot_calibration_curve(test_model_predictions, test_confidences, test_true_labels, 
                           save_path=os.path.join(cfg.XAI_SAVE_DIR, 'reliability_diagram.png'))
    plot_roc_curve(test_model_predictions, test_confidences, test_true_labels,
                   save_path=os.path.join(cfg.XAI_SAVE_DIR, 'roc_curve.png'))
    plot_precision_recall_curve(test_model_predictions, test_confidences, test_true_labels,
                                save_path=os.path.join(cfg.XAI_SAVE_DIR, 'precision_recall_curve.png'))
    plot_confusion_matrix(test_model_predictions, test_true_labels, save_path=os.path.join(cfg.XAI_SAVE_DIR, 'confusion_matrix_overall.png'))

    print("\n--- Quy trình phân loại có chọn lọc hoàn tất ---")


# Chạy pipeline
if __name__ == '__main__':
    run_selective_classification_pipeline(cfg)



