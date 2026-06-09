import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import requests # 保留以备未来可能需要下载
import zipfile
import shutil

# --- 中文字体设置 ---
print("开始设置 Matplotlib 中文字体 (V5 - 使用上传字体)...")

font_file_name = "SIMHEI.TTF" # 从你的日志看，似乎是这个文件名

font_dataset_name = 'simhei'
font_file_uploaded_path = f"/kaggle/input/{font_dataset_name}/{font_file_name}"

# 目标用户字体目录 (Kaggle中通常可写)
user_font_dir = os.path.join(os.path.expanduser('~'), '.local/share/fonts')
font_file_dest_path = os.path.join(user_font_dir, font_file_name)

font_installed_and_setup = False

# 2. 检查上传的字体文件是否存在
if not os.path.exists(font_file_uploaded_path):
    print(f"错误: 未在指定路径找到上传的字体文件: {font_file_uploaded_path}")
    print("请确认:")
    print(f"  1. 你已经将名为 '{font_file_name}' 的字体文件上传为一个 Kaggle Dataset。")
    print(f"  2. 你已经将该 Dataset ('{font_dataset_name}') 添加到了这个 Notebook 的输入中。")
    print(f"  3. 代码中的 `font_dataset_name` 已设置为 '{font_dataset_name}' (如果不是，请修改)。")
else:
    print(f"找到上传的字体文件: {font_file_uploaded_path}")
    try:
        # --- 将字体复制到系统可识别的位置 ---
        os.makedirs(user_font_dir, exist_ok=True)
        if not os.path.exists(font_file_dest_path):
            shutil.copy(font_file_uploaded_path, font_file_dest_path)
            print(f"字体已复制到: {font_file_dest_path}")
        else:
            print(f"字体已存在于目标目录: {font_file_dest_path}")

        # --- 添加字体到 Matplotlib 管理器 ---
        # 使用字体文件的完整路径来添加
        font_entry = fm.FontEntry(fname=font_file_dest_path, name=os.path.splitext(font_file_name)[0]) # 使用文件名作为字体名
        existing_font_paths = [f.fname for f in fm.fontManager.ttflist]
        if font_file_dest_path not in existing_font_paths:
            fm.fontManager.addfont(font_file_dest_path)
            print(f"字体 {font_file_dest_path} 已添加到 FontManager")
        else:
             print(f"字体 {font_file_dest_path} 已在 FontManager 的列表 Ttflist 中。")

        font_installed_and_setup = True # 标记字体文件已就位

    except Exception as e_copy_add:
        print(f"复制或添加字体时出错: {e_copy_add}")


# 3. 如果字体文件在目标位置，则清理缓存并设置rcParams
if font_installed_and_setup: # 仅当字体文件已复制/存在于目标位置时执行
    try:
        # --- 清理缓存 (使用正确的函数 matplotlib.get_cachedir()) ---
        try:
            cache_dir = matplotlib.get_cachedir() # <--- 使用 matplotlib.get_cachedir()
            cache_cleaned = False
            if os.path.exists(cache_dir):
                print(f"尝试清理 Matplotlib 字体缓存目录: {cache_dir}")
                for file in os.listdir(cache_dir):
                    # 匹配更通用的缓存文件名模式
                    if file.startswith('fontlist') and file.endswith(('.json', '.cache', '.afm', '.pickle')):
                        try:
                            os.remove(os.path.join(cache_dir, file))
                            print(f"  已删除缓存文件: {file}")
                            cache_cleaned = True
                        except Exception as e_rm:
                            print(f"  删除缓存文件 {file} 失败: {e_rm}")
                if cache_cleaned:
                     print("Matplotlib 字体缓存文件已清理。可能需要重启 Kernel 使其完全生效。")
                else:
                     print("  未找到需要清理的字体缓存文件。")
            else:
                 print("未找到 Matplotlib 缓存目录。")
        except AttributeError:
             # 如果连 matplotlib.get_cachedir() 都没有 (极旧版本?)，则跳过
             print("警告: 无法使用 matplotlib.get_cachedir()。跳过缓存清理。")
        except Exception as e_cache:
             print(f"清理字体缓存时出错: {e_cache}")


        # --- 设置 Matplotlib 参数 ---
        # 尝试从字体文件获取标准字体名 (如 'SimHei')
        try:
            font_prop = fm.FontProperties(fname=font_file_dest_path)
            font_name = font_prop.get_name()
            print(f"从文件推断出的字体名称: {font_name}")
        except Exception:
            # 如果失败，则使用文件名（不含扩展名）作为备选
            font_name = os.path.splitext(font_file_name)[0]
            print(f"无法从文件获取字体名，使用文件名作为名称: {font_name}")

        # 设置 matplotlib 默认字体
        plt.rcParams['font.family'] = 'sans-serif' # 设置通用族
        plt.rcParams['font.sans-serif'] = [font_name, 'sans-serif'] # 将你的字体名加入列表首位
        plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
        print(f"Matplotlib RCParams 已设置为优先使用 '{font_name}' 显示中文。")

        # 验证一下字体是否被正确识别（可选）
        # if font_name in fm.findSystemFonts(fontpaths=[user_font_dir]):
        #      print(f"验证：字体 '{font_name}' 在管理器中找到。")
        # else:
        #      print(f"警告：字体 '{font_name}' 可能未被管理器完全识别，如果绘图仍有问题请重启Kernel。")


    except Exception as e_setup:
        print(f"设置 Matplotlib 字体参数或清理缓存时出错: {e_setup}")
        font_installed_and_setup = False # 标记设置失败
else:
     # 如果前面步骤失败，提醒用户
     if not os.path.exists(font_file_uploaded_path):
        print("错误：未找到上传的字体文件，无法继续设置。")
     else:
        print("字体复制或添加到管理器时失败，中文可能无法正确显示。")


# --- 结束字体设置 ---
print("-" * 30)


# 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import random
import glob
from tqdm.notebook import tqdm
import pickle
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置路径
DATA_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_2024.csv')
TRAIN_META_PATH = os.path.join(DATA_DIR, 'train_series_meta.csv')
SEGMENTATION_DIR = '/kaggle/input/unet-cache/segmentation_predictions_multi_v2'  # 您生成的分割掩码路径

# 创建输出目录
OUTPUT_DIR = '/kaggle/working/model_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从分割模型继承的环境变量
IMG_SIZE = (224, 224)
TARGET_SIZE = IMG_SIZE[0]
TARGET_SIZE_INT = IMG_SIZE[0]
N_INPUT_CHANNELS = 3
BEST_NIFTI_ORIENTATION_TRANSFORM = None  # 不需要旋转
USE_REVERSE_NIFTI_MAPPING = False  # 使用正向映射

# 训练参数
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_SLICES = 16  # 论文中使用32个切片
NUM_FOLDS = 5

print(f"图像尺寸设置为: {IMG_SIZE}")
print(f"NIFTI方向变换: {BEST_NIFTI_ORIENTATION_TRANSFORM}")
print(f"使用反向映射: {USE_REVERSE_NIFTI_MAPPING}")
print("环境和参数设置完成")


# 加载DICOM切片
def load_dicom_slice(path):
    """加载单个DICOM切片，应用VOI LUT，归一化"""
    try:
        dicom_file = pydicom.dcmread(path)
        instance_number = int(dicom_file.InstanceNumber)
        image = apply_voi_lut(dicom_file.pixel_array, dicom_file)

        # 转换为Hounsfield单位
        intercept = dicom_file.RescaleIntercept
        slope = dicom_file.RescaleSlope
        image = image * slope + intercept

        # 窗口化处理
        window_center = 50  # 腹窗
        window_width = 350
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image = image.copy()
        image[image < img_min] = img_min
        image[image > img_max] = img_max

        # 归一化
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        # 处理 MONOCHROME1 (图像像素值需要反转)
        if 'PhotometricInterpretation' in dicom_file and dicom_file.PhotometricInterpretation == "MONOCHROME1":
            image = 1.0 - image

        return image, instance_number, dicom_file
    except Exception as e:
        # print(f"加载 DICOM 错误 {path}: {e}")
        return None, None, None

# 获取DICOM文件列表
def get_dicom_files_dict(patient_id):
    """获取患者的DICOM文件列表"""
    dicom_info = {}
    patient_dir = os.path.join(TRAIN_IMAGES_DIR, str(patient_id))
    
    if not os.path.exists(patient_dir):
        return dicom_info
        
    for series_id in os.listdir(patient_dir):
        series_dir = os.path.join(patient_dir, series_id)
        if not os.path.isdir(series_dir):
            continue
            
        dicom_files = glob.glob(os.path.join(series_dir, '*.dcm'))
        dicom_tuples = []
        
        for f_path in dicom_files:
            try:
                ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                dicom_tuples.append((int(ds.InstanceNumber), f_path))
            except:
                pass
                
        # 按InstanceNumber排序
        dicom_tuples.sort(key=lambda x: x[0])
        dicom_info[series_id] = dicom_tuples
        
    return dicom_info

# 从分割掩码文件加载掩码
def load_segmentation_mask(mask_path, target_size=TARGET_SIZE_INT):
    """加载分割掩码"""
    try:
        mask_data = np.load(mask_path)
        mask = mask_data['mask']
        
        # 调整大小
        if mask.shape[0] != target_size or mask.shape[1] != target_size:
            resized_mask = np.zeros((target_size, target_size, mask.shape[2]), dtype=mask.dtype)
            for c in range(mask.shape[2]):
                resized_mask[:, :, c] = cv2.resize(mask[:, :, c], (target_size, target_size), 
                                                  interpolation=cv2.INTER_NEAREST)
            mask = resized_mask
            
        return mask
    except Exception as e:
        # print(f"加载掩码错误 {mask_path}: {e}")
        return None

# 增强外渗特征
def enhance_extravasation_features(image, dicom_file, aortic_hu=None):
    """基于CT值特性增强外渗特征"""
    if dicom_file is None:
        return np.zeros_like(image)
        
    # 获取原始HU值
    pixel_array = dicom_file.pixel_array
    intercept = dicom_file.RescaleIntercept
    slope = dicom_file.RescaleSlope
    hu_image = pixel_array * slope + intercept
    
    # 设置阈值
    if aortic_hu is not None and aortic_hu > 100:  # 确保是造影检查
        lower_threshold = aortic_hu * 0.6
        upper_threshold = aortic_hu * 1.1
    else:
        # 默认值
        lower_threshold = 100
        upper_threshold = 300
    
    # 创建掩码
    extravasation_mask = np.zeros_like(hu_image)
    extravasation_mask[(hu_image >= lower_threshold) & (hu_image <= upper_threshold)] = 1
    
    # 形态学操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    extravasation_mask = cv2.morphologyEx(extravasation_mask, cv2.MORPH_OPEN, kernel)
    extravasation_mask = cv2.morphologyEx(extravasation_mask, cv2.MORPH_CLOSE, kernel)
    
    # 调整大小以匹配图像
    if extravasation_mask.shape != image.shape:
        extravasation_mask = cv2.resize(extravasation_mask, (image.shape[1], image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
    
    return extravasation_mask

# 选择代表性切片
def select_representative_slices(slices_data, num_slices=NUM_SLICES, organs_of_interest=['liver', 'spleen', 'kidney', 'bowel']):
    """选择代表性切片，确保每个器官至少出现在一定数量的切片中"""
    if len(slices_data) <= num_slices:
        return slices_data
        
    # 计算每个切片中各器官的存在情况
    organ_presence = []
    for slice_data in slices_data:
        mask = slice_data.get('mask')
        presence = {organ: False for organ in organs_of_interest}
        
        if mask is not None:
            for i, organ in enumerate(organs_of_interest):
                if i < mask.shape[2] and np.any(mask[:, :, i] > 0.5):
                    presence[organ] = True
                    
        organ_presence.append(presence)
    
    # 分数计算：每个切片覆盖的器官数量
    slice_scores = []
    for i, presence in enumerate(organ_presence):
        score = sum(presence.values())
        slice_scores.append((i, score))
    
    # 按分数降序排序
    slice_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 选择分数最高的切片
    selected_indices = [score[0] for score in slice_scores[:num_slices]]
    selected_indices.sort()  # 按原始顺序排序
    
    return [slices_data[i] for i in selected_indices]

# 图像预处理和调整大小
def preprocess_image(image, target_size=TARGET_SIZE_INT):
    """预处理图像，调整大小"""
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image


class AbdominalTraumaDataset(Dataset):
    def __init__(self, patient_ids, meta_df, transform=None, num_slices=NUM_SLICES, mode='train'):
        self.patient_ids = patient_ids
        self.meta_df = meta_df
        self.transform = transform
        self.num_slices = num_slices
        self.mode = mode
        self.labels_df = pd.read_csv(TRAIN_CSV_PATH)

        self.aortic_hu_map = {}
        for _, row in meta_df.iterrows():
            patient_id = str(row['patient_id'])
            series_id = str(row['series_id'])
            aortic_hu = row['aortic_hu']
            if patient_id not in self.aortic_hu_map:
                self.aortic_hu_map[patient_id] = {}
            self.aortic_hu_map[patient_id][series_id] = aortic_hu

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = str(self.patient_ids[idx]) # 确保 patient_id 是字符串
        dicom_files_dict = get_dicom_files_dict(patient_id)

        if not dicom_files_dict:
            return self._create_empty_sample()

        series_id, dicom_tuples = next(iter(dicom_files_dict.items()))

        aortic_hu = 200.0 # 默认值
        if patient_id in self.aortic_hu_map and series_id in self.aortic_hu_map[patient_id]:
            aortic_hu = self.aortic_hu_map[patient_id][series_id]
            if pd.isna(aortic_hu): # 处理可能的 NaN 值
                 aortic_hu = 200.0

        slices_data = []
        for dicom_idx, (instance_number, dicom_path) in enumerate(dicom_tuples):
            raw_dicom_image, _, dicom_file_obj = load_dicom_slice(dicom_path)

            if raw_dicom_image is None:
                continue

            current_extravasation_features_orig_size = enhance_extravasation_features(raw_dicom_image, dicom_file_obj, aortic_hu)
            processed_main_image = preprocess_image(raw_dicom_image.copy(), target_size=TARGET_SIZE_INT)

            mask_path = os.path.join(SEGMENTATION_DIR, patient_id, series_id, f"{instance_number}.npz")
            mask = None
            if os.path.exists(mask_path):
                mask = load_segmentation_mask(mask_path, target_size=TARGET_SIZE_INT)

            if current_extravasation_features_orig_size.shape[0] != TARGET_SIZE_INT or \
               current_extravasation_features_orig_size.shape[1] != TARGET_SIZE_INT:
                resized_extravasation_features = cv2.resize(
                    current_extravasation_features_orig_size,
                    (TARGET_SIZE_INT, TARGET_SIZE_INT),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                resized_extravasation_features = current_extravasation_features_orig_size

            slices_data.append({
                'instance_number': instance_number,
                'image': processed_main_image,
                'mask': mask,
                'extravasation_features': resized_extravasation_features
            })

        if not slices_data: # 如果所有切片都无法加载
            return self._create_empty_sample()

        if len(slices_data) > self.num_slices:
            slices_data = select_representative_slices(slices_data, self.num_slices)

        while len(slices_data) < self.num_slices:
            if slices_data: # 确保 slices_data 不为空
                slices_data.append(slices_data[-1])
            else: # 理论上不应该执行到这里，因为前面有 if not slices_data: return ...
                return self._create_empty_sample()


        images_list = []
        masks_list = []
        extravasation_features_list = []

        for slice_data in slices_data:
            img_np = slice_data['image'] # 已经是 numpy array (H, W)
            if self.transform:
                # ToPILImage 需要 (H, W) 或 (H, W, C)
                # 如果 img_np 是 (H,W)，ToPILImage 会将其视为 'L' 模式
                # ToTensor 会将 PIL 'L' 模式图像转换为 [1, H, W]
                img_tensor = self.transform(img_np)
            else:
                img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) # [1, H, W]
            images_list.append(img_tensor)

            if slice_data['mask'] is not None:
                masks_list.append(slice_data['mask']) # (H, W, 4)
            else:
                empty_mask = np.zeros((TARGET_SIZE_INT, TARGET_SIZE_INT, 4), dtype=np.float32)
                masks_list.append(empty_mask)

            extravasation_features_list.append(slice_data['extravasation_features']) # (H, W)

        images = torch.stack(images_list).float() # [num_slices, 1, H, W]
        # masks_list 是 [(H,W,4), (H,W,4), ...]
        # np.stack(masks_list) -> [num_slices, H, W, 4]
        masks = torch.tensor(np.stack(masks_list), dtype=torch.float32)
        # extravasation_features_list 是 [(H,W), (H,W), ...]
        # np.stack(extravasation_features_list) -> [num_slices, H, W]
        # .unsqueeze(1) -> [num_slices, 1, H, W]
        extravasation_features_tensor = torch.tensor(np.stack(extravasation_features_list), dtype=torch.float32).unsqueeze(1)
        aortic_hu_tensor = torch.tensor([float(aortic_hu)], dtype=torch.float32)


        if self.mode == 'train' or self.mode == 'val':
            # patient_id 在这里应该是 int 类型以便于在 labels_df 中查找
            label_row = self.labels_df[self.labels_df['patient_id'] == int(patient_id)].iloc[0]
            bowel_label = int(label_row['bowel_healthy'] == 0)
            extravasation_label = int(label_row['extravasation_healthy'] == 0)
            kidney_label = int(label_row['kidney_healthy'] == 0) * (1 + int(label_row['kidney_low'] == 0))
            liver_label = int(label_row['liver_healthy'] == 0) * (1 + int(label_row['liver_low'] == 0))
            spleen_label = int(label_row['spleen_healthy'] == 0) * (1 + int(label_row['spleen_low'] == 0))
        else: # test mode
            bowel_label = 0
            extravasation_label = 0
            kidney_label = 0
            liver_label = 0
            spleen_label = 0

        return {
            'patient_id': patient_id, # 返回原始的 patient_id (str)
            'images': images,
            'masks': masks,
            'extravasation_features': extravasation_features_tensor,
            'aortic_hu': aortic_hu_tensor,
            'bowel': torch.tensor([bowel_label], dtype=torch.float32),
            'extravasation': torch.tensor([extravasation_label], dtype=torch.float32),
            'kidney': torch.tensor(kidney_label, dtype=torch.long),
            'liver': torch.tensor(liver_label, dtype=torch.long),
            'spleen': torch.tensor(spleen_label, dtype=torch.long)
        }

    def _create_empty_sample(self):
        empty_images = torch.zeros((self.num_slices, 1, TARGET_SIZE_INT, TARGET_SIZE_INT), dtype=torch.float32)
        empty_masks = torch.zeros((self.num_slices, TARGET_SIZE_INT, TARGET_SIZE_INT, 4), dtype=torch.float32)
        empty_extravasation = torch.zeros((self.num_slices, 1, TARGET_SIZE_INT, TARGET_SIZE_INT), dtype=torch.float32)
        empty_aortic_hu = torch.tensor([0.0], dtype=torch.float32)

        return {
            'patient_id': '0', # 虚拟 patient_id
            'images': empty_images,
            'masks': empty_masks,
            'extravasation_features': empty_extravasation,
            'aortic_hu': empty_aortic_hu,
            'bowel': torch.tensor([0], dtype=torch.float32),
            'extravasation': torch.tensor([0], dtype=torch.float32),
            'kidney': torch.tensor(0, dtype=torch.long),
            'liver': torch.tensor(0, dtype=torch.long),
            'spleen': torch.tensor(0, dtype=torch.long)
        }



from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter, GaussianBlur, ToTensor, ToPILImage
import PIL.Image as Image

# 定义数据增强
def get_transforms(mode='train'):
    if mode == 'train':
        return Compose([
            ToPILImage(),  # 先将NumPy数组转换为PIL Image
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ColorJitter(brightness=0.2, contrast=0.2),
            GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ToTensor(),  # 转换回Tensor
        ])
    else:
        return ToTensor()  # 验证集只需要转换为Tensor

# 准备数据集
def prepare_data(meta_df, num_folds=NUM_FOLDS, fold_idx=0):
    # 获取所有患者ID
    all_patients = meta_df['patient_id'].unique()
    
    # 创建KFold分割
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    folds = list(kf.split(all_patients))
    
    train_indices, val_indices = folds[fold_idx]
    train_patients = [str(all_patients[i]) for i in train_indices]
    val_patients = [str(all_patients[i]) for i in val_indices]
    
    # 创建数据集
    train_dataset = AbdominalTraumaDataset(
        train_patients, meta_df, transform=get_transforms('train'), mode='train'
    )
    val_dataset = AbdominalTraumaDataset(
        val_patients, meta_df, transform=get_transforms('val'), mode='train'
    )
    
    # 创建数据加载器 - 减少worker数量以避免潜在的多进程问题
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader

# 加载元数据
def load_meta_data():
    meta_df = pd.read_csv(TRAIN_META_PATH)
    # 确保数据类型正确
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    meta_df['series_id'] = meta_df['series_id'].astype(str)
    return meta_df

print("数据加载和增强函数定义完成")


class AbdominalTraumaModel(nn.Module):
    """腹部外伤2.5D分类模型，实现论文中描述的架构"""
    
    def __init__(self, num_slices=NUM_SLICES):
        super().__init__()
        
        # EfficientNetB1作为特征提取器
        self.efficientnet = models.efficientnet_b1(pretrained=True)
        self.feature_dim = 1280  # EfficientNetB1的特征维度
        
        # 掩码处理分支
        self.mask_conv = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.mask_pool = nn.AdaptiveAvgPool2d(1)
        self.mask_fc = nn.Linear(16, 64)
        
        # 外渗特征处理分支
        self.extra_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.extra_pool = nn.AdaptiveAvgPool2d(1)
        self.extra_fc = nn.Linear(8, 32)
        
        # aortic_hu处理
        self.aortic_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # LSTM处理时序特征
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Neck结构
        self.neck = nn.Sequential(
            nn.Linear(512*2, 512),  # 双向LSTM，所以是512*2
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 多标签分类头
        combined_features = 256 + 64 + 32 + 32  # Neck + 掩码特征 + 外渗特征 + aortic_hu特征
        self.bowel_head = nn.Linear(combined_features, 1)
        self.extravasation_head = nn.Linear(combined_features, 1)
        self.kidney_head = nn.Linear(combined_features, 3)
        self.liver_head = nn.Linear(combined_features, 3)
        self.spleen_head = nn.Linear(combined_features, 3)
    
    def forward(self, images, masks, extravasation_features, aortic_hu):
        """
        参数:
            images: [B, num_slices, C, H, W] - 批次中每个患者的切片，C可能是1或3
            masks: [B, num_slices, H, W, 4] - 对应的分割掩码
            extravasation_features: [B, num_slices, 1, H, W] - 外渗特征
            aortic_hu: [B, 1] - 主动脉HU值
        返回:
            各器官的损伤预测结果
        """
        batch_size = images.size(0)
        seq_len = images.size(1)  # 应该是num_slices
        
        # 检查输入通道数，确保是3通道
        if images.size(2) == 1:  # 如果是单通道
            # 重塑以便批量处理所有切片
            images_reshaped = images.view(batch_size * seq_len, 1, images.size(3), images.size(4))
            # 扩展为3通道
            images_reshaped = images_reshaped.repeat(1, 3, 1, 1)
        else:  # 已经是3通道
            images_reshaped = images.view(batch_size * seq_len, images.size(2), images.size(3), images.size(4))
        
        # 处理掩码
        masks_reshaped = masks.view(batch_size * seq_len, masks.size(2), masks.size(3), masks.size(4))
        masks_reshaped = masks_reshaped.permute(0, 3, 1, 2)  # [B*seq_len, 4, H, W]
        
        # 处理外渗特征
        extra_reshaped = extravasation_features.view(batch_size * seq_len, 1, extravasation_features.size(3), extravasation_features.size(4))
        
        # 特征提取
        features = self.efficientnet.features(images_reshaped)  # [B*seq_len, 1280, h, w]
        
        # 掩码特征处理
        mask_features = self.mask_conv(masks_reshaped)  # [B*seq_len, 16, h, w]
        mask_features = self.mask_pool(mask_features).squeeze(-1).squeeze(-1)  # [B*seq_len, 16]
        mask_features = self.mask_fc(mask_features)  # [B*seq_len, 64]
        
        # 外渗特征处理
        extra_features = self.extra_conv(extra_reshaped)  # [B*seq_len, 8, h, w]
        extra_features = self.extra_pool(extra_features).squeeze(-1).squeeze(-1)  # [B*seq_len, 8]
        extra_features = self.extra_fc(extra_features)  # [B*seq_len, 32]
        
        # 全局平均池化EfficientNet特征
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # [B*seq_len, 1280]
        
        # 重塑回序列形式
        sequence_features = pooled_features.view(batch_size, seq_len, -1)  # [B, seq_len, 1280]
        
        # LSTM处理序列特征
        lstm_out, _ = self.lstm(sequence_features)  # [B, seq_len, 512*2]
        
        # 取最后一个时间步的输出
        final_lstm_out = lstm_out[:, -1, :]  # [B, 512*2]
        
        # Neck结构处理
        neck_out = self.neck(final_lstm_out)  # [B, 256]
        
        # 重塑掩码特征和外渗特征以匹配批次大小
        mask_features = mask_features.view(batch_size, seq_len, -1)  # [B, seq_len, 64]
        mask_features_avg = torch.mean(mask_features, dim=1)  # [B, 64]
        
        extra_features = extra_features.view(batch_size, seq_len, -1)  # [B, seq_len, 32]
        extra_features_avg = torch.mean(extra_features, dim=1)  # [B, 32]
        
        # 处理aortic_hu
        aortic_features = self.aortic_encoder(aortic_hu)  # [B, 32]
        
        # 连接特征
        combined_features = torch.cat([neck_out, mask_features_avg, extra_features_avg, aortic_features], dim=1)  # [B, 256+64+32+32]
        
        # 多标签分类
        bowel_out = self.bowel_head(combined_features)
        extravasation_out = self.extravasation_head(combined_features)
        kidney_out = self.kidney_head(combined_features)
        liver_out = self.liver_head(combined_features)
        spleen_out = self.spleen_head(combined_features)
        
        return {
            'bowel': bowel_out,
            'extravasation': extravasation_out,
            'kidney': kidney_out,
            'liver': liver_out,
            'spleen': spleen_out
        }

# 初始化模型
model = AbdominalTraumaModel().to(device)
print("模型定义完成")


# 定义损失函数
bce_bowel = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
bce_extravasation = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.05, weight=torch.tensor([1.0, 2.0, 4.0]).to(device))

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True
)

# 指标计算
class MetricsCalculator:
    def __init__(self, mode='binary'):
        self.probabilities = []
        self.predictions = []
        self.targets = []
        self.mode = mode
    
    def update(self, logits, target):
        if self.mode == 'binary':
            probabilities = torch.sigmoid(logits)
            predicted = (probabilities > 0.5)
        else:
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            
        self.probabilities.extend(probabilities.detach().cpu().numpy())
        self.predictions.extend(predicted.detach().cpu().numpy())
        self.targets.extend(target.detach().cpu().numpy())
    
    def reset(self):
        self.probabilities = []
        self.predictions = []
        self.targets = []
    
    def compute_accuracy(self):
        if not self.predictions:
            return 0.0
        return np.mean(np.array(self.predictions) == np.array(self.targets))
    
    def compute_auc(self):
        if not self.probabilities or len(np.unique(self.targets)) < 2:
            return 0.0
        try:
            if self.mode == 'multi':
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(self.targets, self.probabilities, multi_class='ovo')
            else:
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(self.targets, self.probabilities)
        except:
            return 0.0

# 初始化指标计算器
train_metrics = {
    'bowel': MetricsCalculator('binary'),
    'extravasation': MetricsCalculator('binary'),
    'kidney': MetricsCalculator('multi'),
    'liver': MetricsCalculator('multi'),
    'spleen': MetricsCalculator('multi')
}

val_metrics = {
    'bowel': MetricsCalculator('binary'),
    'extravasation': MetricsCalculator('binary'),
    'kidney': MetricsCalculator('multi'),
    'liver': MetricsCalculator('multi'),
    'spleen': MetricsCalculator('multi')
}

print("损失函数和优化器设置完成")


def train_epoch(model, train_loader, optimizer, metrics):
    """训练单个epoch"""
    model.train()
    total_loss = 0
    
    # 重置指标
    for metric in metrics.values():
        metric.reset()
    
    # 进度条
    progress_bar = tqdm(train_loader, desc="训练")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 获取数据
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)
        extravasation_features = batch['extravasation_features'].to(device)
        aortic_hu = batch['aortic_hu'].to(device)
        
        # 获取标签
        bowel = batch['bowel'].to(device)
        extravasation = batch['extravasation'].to(device)
        kidney = batch['kidney'].to(device)
        liver = batch['liver'].to(device)
        spleen = batch['spleen'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images, masks, extravasation_features, aortic_hu)
        
        # 计算损失
        bowel_loss = bce_bowel(outputs['bowel'], bowel)
        extravasation_loss = bce_extravasation(outputs['extravasation'], extravasation)
        kidney_loss = ce_loss(outputs['kidney'], kidney)
        liver_loss = ce_loss(outputs['liver'], liver)
        spleen_loss = ce_loss(outputs['spleen'], spleen)
        
        # 总损失
        loss = bowel_loss + extravasation_loss + kidney_loss + liver_loss + spleen_loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新总损失
        total_loss += loss.item()
        
        # 更新指标
        metrics['bowel'].update(outputs['bowel'], bowel)
        metrics['extravasation'].update(outputs['extravasation'], extravasation)
        metrics['kidney'].update(outputs['kidney'], kidney)
        metrics['liver'].update(outputs['liver'], liver)
        metrics['spleen'].update(outputs['spleen'], spleen)
        
        # 更新进度条
        progress_bar.set_postfix(loss=loss.item())
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    
    # 计算指标
    metrics_results = {
        'loss': avg_loss,
        'bowel_acc': metrics['bowel'].compute_accuracy(),
        'extravasation_acc': metrics['extravasation'].compute_accuracy(),
        'kidney_acc': metrics['kidney'].compute_accuracy(),
        'liver_acc': metrics['liver'].compute_accuracy(),
        'spleen_acc': metrics['spleen'].compute_accuracy(),
        'bowel_auc': metrics['bowel'].compute_auc(),
        'extravasation_auc': metrics['extravasation'].compute_auc(),
        'kidney_auc': metrics['kidney'].compute_auc(),
        'liver_auc': metrics['liver'].compute_auc(),
        'spleen_auc': metrics['spleen'].compute_auc()
    }
    
    return metrics_results

def validate(model, val_loader, metrics):
    """验证模型"""
    model.eval()
    total_loss = 0
    
    # 重置指标
    for metric in metrics.values():
        metric.reset()
    
    # 进度条
    progress_bar = tqdm(val_loader, desc="验证")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            extravasation_features = batch['extravasation_features'].to(device)
            aortic_hu = batch['aortic_hu'].to(device)
            
            # 获取标签
            bowel = batch['bowel'].to(device)
            extravasation = batch['extravasation'].to(device)
            kidney = batch['kidney'].to(device)
            liver = batch['liver'].to(device)
            spleen = batch['spleen'].to(device)
            
            # 前向传播
            outputs = model(images, masks, extravasation_features, aortic_hu)
            
            # 计算损失
            bowel_loss = bce_bowel(outputs['bowel'], bowel)
            extravasation_loss = bce_extravasation(outputs['extravasation'], extravasation)
            kidney_loss = ce_loss(outputs['kidney'], kidney)
            liver_loss = ce_loss(outputs['liver'], liver)
            spleen_loss = ce_loss(outputs['spleen'], spleen)
            
            # 总损失
            loss = bowel_loss + extravasation_loss + kidney_loss + liver_loss + spleen_loss
            
            # 更新总损失
            total_loss += loss.item()
            
            # 更新指标
            metrics['bowel'].update(outputs['bowel'], bowel)
            metrics['extravasation'].update(outputs['extravasation'], extravasation)
            metrics['kidney'].update(outputs['kidney'], kidney)
            metrics['liver'].update(outputs['liver'], liver)
            metrics['spleen'].update(outputs['spleen'], spleen)

            # 打印每步信息
            if (batch_idx + 1) % 10 == 0:  # 每10步打印一次，可以根据需要调整
                print(f"步骤 [{batch_idx+1}/{len(train_loader)}] 损失: {loss.item():.4f}")
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 计算指标
    metrics_results = {
        'loss': avg_loss,
        'bowel_acc': metrics['bowel'].compute_accuracy(),
        'extravasation_acc': metrics['extravasation'].compute_accuracy(),
        'kidney_acc': metrics['kidney'].compute_accuracy(),
        'liver_acc': metrics['liver'].compute_accuracy(),
        'spleen_acc': metrics['spleen'].compute_accuracy(),
        'bowel_auc': metrics['bowel'].compute_auc(),
        'extravasation_auc': metrics['extravasation'].compute_auc(),
        'kidney_auc': metrics['kidney'].compute_auc(),
        'liver_auc': metrics['liver'].compute_auc(),
        'spleen_auc': metrics['spleen'].compute_auc()
    }
    
    return metrics_results

print("训练和验证函数定义完成")


# 加载元数据
meta_df = load_meta_data()

# 记录训练历史
history = {
    'train_loss': [],
    'val_loss': [],
    'train_metrics': [],
    'val_metrics': []
}

# 最佳验证损失
best_val_loss = float('inf')

# 使用多折交叉验证
for fold in range(NUM_FOLDS):
    print(f"\n===== 开始训练折 {fold+1}/{NUM_FOLDS} =====")
    
    # 准备数据
    train_loader, val_loader = prepare_data(meta_df, NUM_FOLDS, fold)
    
    # 重新初始化模型
    if fold > 0:
        model = AbdominalTraumaModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # 训练
        train_metrics_results = train_epoch(model, train_loader, optimizer, train_metrics)
        history['train_loss'].append(train_metrics_results['loss'])
        history['train_metrics'].append(train_metrics_results)
        
        # 验证
        val_metrics_results = validate(model, val_loader, val_metrics)
        history['val_loss'].append(val_metrics_results['loss'])
        history['val_metrics'].append(val_metrics_results)
        
        # 打印结果
        print(f"训练损失: {train_metrics_results['loss']:.4f}, 验证损失: {val_metrics_results['loss']:.4f}")
        print(f"训练 AUC - 肠道: {train_metrics_results['bowel_auc']:.4f}, 外渗: {train_metrics_results['extravasation_auc']:.4f}, "
              f"肾脏: {train_metrics_results['kidney_auc']:.4f}, 肝脏: {train_metrics_results['liver_auc']:.4f}, "
              f"脾脏: {train_metrics_results['spleen_auc']:.4f}")
        print(f"验证 AUC - 肠道: {val_metrics_results['bowel_auc']:.4f}, 外渗: {val_metrics_results['extravasation_auc']:.4f}, "
              f"肾脏: {val_metrics_results['kidney_auc']:.4f}, 肝脏: {val_metrics_results['liver_auc']:.4f}, "
              f"脾脏: {val_metrics_results['spleen_auc']:.4f}")
        
        # 更新学习率
        scheduler.step(val_metrics_results['loss'])
        
        # 保存最佳模型
        if val_metrics_results['loss'] < best_val_loss:
            best_val_loss = val_metrics_results['loss']
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_model_fold{fold}.pt'))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'final_model_fold{fold}.pt'))
    print(f"折 {fold+1} 训练完成，保存最终模型")

print("\n训练完成！")


# 绘制损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='训练损失')
plt.plot(history['val_loss'], label='验证损失')
plt.title('训练和验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()

# 绘制AUC曲线
plt.subplot(1, 2, 2)
train_bowel_auc = [m['bowel_auc'] for m in history['train_metrics']]
train_extravasation_auc = [m['extravasation_auc'] for m in history['train_metrics']]
train_liver_auc = [m['liver_auc'] for m in history['train_metrics']]
train_kidney_auc = [m['kidney_auc'] for m in history['train_metrics']]
train_spleen_auc = [m['spleen_auc'] for m in history['train_metrics']]

val_bowel_auc = [m['bowel_auc'] for m in history['val_metrics']]
val_extravasation_auc = [m['extravasation_auc'] for m in history['val_metrics']]
val_liver_auc = [m['liver_auc'] for m in history['val_metrics']]
val_kidney_auc = [m['kidney_auc'] for m in history['val_metrics']]
val_spleen_auc = [m['spleen_auc'] for m in history['val_metrics']]

plt.plot(val_bowel_auc, label='肠道')
plt.plot(val_extravasation_auc, label='外渗')
plt.plot(val_liver_auc, label='肝脏')
plt.plot(val_kidney_auc, label='肾脏')
plt.plot(val_spleen_auc, label='脾脏')
plt.title('验证集各器官AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
plt.show()

# 打印最终结果表格
from tabulate import tabulate

final_metrics = history['val_metrics'][-1]
metrics_table = [
    ['器官', 'AUC', '准确率'],
    ['肠道', f"{final_metrics['bowel_auc']:.4f}", f"{final_metrics['bowel_acc']:.4f}"],
    ['外渗', f"{final_metrics['extravasation_auc']:.4f}", f"{final_metrics['extravasation_acc']:.4f}"],
    ['肝脏', f"{final_metrics['liver_auc']:.4f}", f"{final_metrics['liver_acc']:.4f}"],
    ['肾脏', f"{final_metrics['kidney_auc']:.4f}", f"{final_metrics['kidney_acc']:.4f}"],
    ['脾脏', f"{final_metrics['spleen_auc']:.4f}", f"{final_metrics['spleen_acc']:.4f}"]
]

print("\n最终验证结果:")
print(tabulate(metrics_table, headers='firstrow', tablefmt='grid'))


def load_best_model():
    """加载最佳模型"""
    best_model = AbdominalTraumaModel().to(device)
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_fold0.pt')
    
    if os.path.exists(best_model_path):
        best_model.load_state_dict(torch.load(best_model_path))
        print(f"加载最佳模型: {best_model_path}")
    else:
        print(f"未找到最佳模型，使用当前模型")
        best_model = model
        
    return best_model

def visualize_predictions(model, val_loader, num_samples=3):
    """可视化模型预测结果"""
    model.eval()
    
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if samples_visualized >= num_samples:
                break
                
            # 获取数据
            patient_id = batch['patient_id'][0]
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            extravasation_features = batch['extravasation_features'].to(device)
            aortic_hu = batch['aortic_hu'].to(device)
            
            # 获取标签
            bowel_gt = batch['bowel'].cpu().numpy()[0][0]
            extravasation_gt = batch['extravasation'].cpu().numpy()[0][0]
            kidney_gt = batch['kidney'].cpu().numpy()[0]
            liver_gt = batch['liver'].cpu().numpy()[0]
            spleen_gt = batch['spleen'].cpu().numpy()[0]
            
            # 前向传播
            outputs = model(images, masks, extravasation_features, aortic_hu)
            
            # 获取预测结果
            bowel_pred = torch.sigmoid(outputs['bowel']).cpu().numpy()[0][0] > 0.5
            extravasation_pred = torch.sigmoid(outputs['extravasation']).cpu().numpy()[0][0] > 0.5
            kidney_pred = torch.argmax(outputs['kidney'], dim=1).cpu().numpy()[0]
            liver_pred = torch.argmax(outputs['liver'], dim=1).cpu().numpy()[0]
            spleen_pred = torch.argmax(outputs['spleen'], dim=1).cpu().numpy()[0]
            
            # 可视化
            plt.figure(figsize=(20, 10))
            
            # 选择4个代表性切片进行显示
            slice_indices = np.linspace(0, images.size(1) - 1, 4, dtype=int)
            
            for i, idx in enumerate(slice_indices):
                plt.subplot(2, 4, i + 1)
                plt.imshow(images[0, idx].cpu().numpy(), cmap='gray')
                plt.title(f'切片 {idx}')
                plt.axis('off')
                
                # 显示掩码叠加
                plt.subplot(2, 4, i + 5)
                
                # 创建RGB掩码
                mask_rgb = np.zeros((images.shape[2], images.shape[3], 3))
                
                # 肝脏 - 红色
                if masks[0, idx, :, :, 0].sum() > 0:
                    mask_rgb[:, :, 0] += masks[0, idx, :, :, 0].cpu().numpy() * 0.5
                
                # 脾脏 - 绿色
                if masks[0, idx, :, :, 1].sum() > 0:
                    mask_rgb[:, :, 1] += masks[0, idx, :, :, 1].cpu().numpy() * 0.5
                
                # 肾脏 - 蓝色
                if masks[0, idx, :, :, 2].sum() > 0:
                    mask_rgb[:, :, 2] += masks[0, idx, :, :, 2].cpu().numpy() * 0.5
                
                # 肠道 - 黄色
                if masks[0, idx, :, :, 3].sum() > 0:
                    mask_rgb[:, :, 0] += masks[0, idx, :, :, 3].cpu().numpy() * 0.5
                    mask_rgb[:, :, 1] += masks[0, idx, :, :, 3].cpu().numpy() * 0.5
                
                # 外渗 - 紫色
                if extravasation_features[0, idx, 0].sum() > 0:
                    mask_rgb[:, :, 0] += extravasation_features[0, idx, 0].cpu().numpy() * 0.5
                    mask_rgb[:, :, 2] += extravasation_features[0, idx, 0].cpu().numpy() * 0.5
                
                # 叠加到原图
                img_rgb = np.stack([images[0, idx].cpu().numpy()] * 3, axis=2)
                overlay = img_rgb * 0.7 + mask_rgb * 0.3
                
                plt.imshow(np.clip(overlay, 0, 1))
                plt.title(f'掩码叠加')
                plt.axis('off')
            
            # 显示预测结果
            plt.suptitle(f'患者 {patient_id} - aortic_hu: {aortic_hu.item():.1f}\n'
                         f'肠道: 真实={bowel_gt}, 预测={bowel_pred} | '
                         f'外渗: 真实={extravasation_gt}, 预测={extravasation_pred} | '
                         f'肾脏: 真实={kidney_gt}, 预测={kidney_pred} | '
                         f'肝脏: 真实={liver_gt}, 预测={liver_pred} | '
                         f'脾脏: 真实={spleen_gt}, 预测={spleen_pred}', 
                         fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_{patient_id}.png'))
            plt.show()
            
            samples_visualized += 1

# 加载最佳模型
best_model = load_best_model()

# 准备验证数据
_, val_loader = prepare_data(meta_df, NUM_FOLDS, 0)

# 可视化预测结果
visualize_predictions(best_model, val_loader, num_samples=3)

