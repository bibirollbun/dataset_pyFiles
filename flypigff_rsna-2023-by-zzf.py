# 导入必要的库
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import pydicom
import random
import glob
import pydicom
import nibabel as nib
from skimage.transform import resize


# 检查GPU是否可用
print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", tf.config.list_physical_devices('GPU'))


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()


# 定义数据路径
DATA_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
TRAIN_CSV = os.path.join(DATA_DIR, 'train_2024.csv')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')
TRAIN_META = os.path.join(DATA_DIR, 'train_series_meta.csv')
OUTPUT_DIR = '/kaggle/working'
SEGMENTATION_DIR = "/kaggle/input/rsna-2023-abdominal-trauma-detection/segmentations"
TRAIN_DICOM_TAGS = '/kaggle/input/rsna-2023-abdominal-trauma-detection/train_dicom_tags.parquet' 

# 图像尺寸统一为论文中的224×224
IMG_SIZE = (224, 224)
NUM_ORGANS = 5  # 肝脏, 脾脏, 肾脏, 肠道, 外渗

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)



# 加载训练标签并处理数据
def load_and_process_data():
    """Load and process training data"""
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Training data shape: {train_df.shape}")
    print(f"Data type of patient_id in train_df: {train_df['patient_id'].dtype}") #Add this line
    print(f"Sample rows from train_df: \n{train_df.head()}") #Add this line
    train_df['patient_id'] = train_df['patient_id'].astype(str)

    # 显示前几行数据
    print("\nTraining data sample:")
    print(train_df.head())

    # 检查列名
    print("\nColumn names:")
    print(train_df.columns.tolist())

    # 处理不同类型的器官标签
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']

    # 创建图表
    plt.figure(figsize=(20, 15))

    # 为每个器官创建饼图
    for i, organ in enumerate(organs):
        plt.subplot(2, 3, i+1)

        # 收集该器官的所有状态数据
        status_data = {}

        # 检查健康状态
        if f'{organ}_healthy' in train_df.columns:
            healthy_count = train_df[f'{organ}_healthy'].sum()
            total = len(train_df)
            status_data['Healthy'] = healthy_count
            print(f"\n{organ.capitalize()} - Healthy: {healthy_count} ({healthy_count/total*100:.2f}%)")

        # 检查损伤状态
        if f'{organ}_injury' in train_df.columns:
            injury_count = train_df[f'{organ}_injury'].sum()
            total = len(train_df)
            status_data['Injury'] = injury_count
            print(f"{organ.capitalize()} - Injury: {injury_count} ({injury_count/total*100:.2f}%)")

        # 检查低度损伤
        if f'{organ}_low' in train_df.columns:
            low_count = train_df[f'{organ}_low'].sum()
            total = len(train_df)
            status_data['Low-grade Injury'] = low_count
            print(f"{organ.capitalize()} - Low-grade Injury: {low_count} ({low_count/total*100:.2f}%)")

        # 检查高度损伤
        if f'{organ}_high' in train_df.columns:
            high_count = train_df[f'{organ}_high'].sum()
            total = len(train_df)
            status_data['High-grade Injury'] = high_count
            print(f"{organ.capitalize()} - High-grade Injury: {high_count} ({high_count/total*100:.2f}%)")

        # 绘制饼图
        if status_data:
            labels = status_data.keys()
            sizes = status_data.values()

            # 计算百分比
            total = sum(sizes)
            sizes_percent = [size/total*100 for size in sizes]

            # 添加百分比到标签
            labels_with_percent = [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes_percent)]

            # 设置颜色
            colors = plt.cm.Paired(np.arange(len(sizes)) / len(sizes))

            # 突出显示损伤部分
            explode = [0] * len(sizes)
            for j, label in enumerate(labels):
                if 'Injury' in label:
                    explode[j] = 0.1

            # 绘制饼图
            plt.pie(sizes, explode=explode, labels=labels_with_percent,
                    colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # 确保饼图是圆的
            plt.title(f'{organ.capitalize()} Status Distribution', fontsize=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'organ_status_distribution.png'), dpi=300)
    plt.show()

    # 绘制器官损伤比例的条形图
    plt.figure(figsize=(12, 8))

    # 收集所有器官的损伤百分比
    organ_injury_percentages = []

    for organ in organs:
        # 计算损伤的总数（包括低度和高度损伤）
        injury_count = 0

        if f'{organ}_injury' in train_df.columns:
            injury_count += train_df[f'{organ}_injury'].sum()

        if f'{organ}_low' in train_df.columns:
            injury_count += train_df[f'{organ}_low'].sum()

        if f'{organ}_high' in train_df.columns:
            injury_count += train_df[f'{organ}_high'].sum()

        total = len(train_df)
        injury_percentage = injury_count / total * 100
        organ_injury_percentages.append(injury_percentage)

    # 绘制条形图
    bars = plt.bar(organs, organ_injury_percentages, color=plt.cm.viridis(np.linspace(0, 1, len(organs))))

    # 添加数值标签
    for bar, percentage in zip(bars, organ_injury_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=12)

    plt.xlabel('Organ', fontsize=14)
    plt.ylabel('Injury Percentage (%)', fontsize=14)
    plt.title('Injury Percentage by Organ', fontsize=16)
    plt.ylim(0, max(organ_injury_percentages) * 1.2)  # 设置y轴上限
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, 'organ_injury_percentages.png'), dpi=300)
    plt.show()

    # 如果数据集中有损伤程度的区分，绘制损伤严重程度分布图
    has_severity_data = any(f'{organ}_low' in train_df.columns or f'{organ}_high' in train_df.columns for organ in organs)

    if has_severity_data:
        plt.figure(figsize=(14, 10))

        # 为每个有严重程度区分的器官创建子图
        severity_organs = [organ for organ in organs if f'{organ}_low' in train_df.columns or f'{organ}_high' in train_df.columns]

        for i, organ in enumerate(severity_organs):
            plt.subplot(2, 3, i+1)

            # 收集该器官的损伤严重程度数据
            severity_data = {}

            if f'{organ}_low' in train_df.columns:
                low_count = train_df[f'{organ}_low'].sum()
                severity_data['Low-grade Injury'] = low_count

            if f'{organ}_high' in train_df.columns:
                high_count = train_df[f'{organ}_high'].sum()
                severity_data['High-grade Injury'] = high_count

            if severity_data:
                # 绘制饼图
                labels = severity_data.keys()
                sizes = severity_data.values()

                # 计算百分比
                total = sum(sizes)
                if total > 0:  # 避免除以零
                    sizes_percent = [size/total*100 for size in sizes]

                    # 添加百分比到标签
                    labels_with_percent = [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes_percent)]

                    # 设置颜色
                    colors = ['#ff9999', '#ff3333']  # 浅红色和深红色分别表示低度和高度损伤

                    # 绘制饼图
                    plt.pie(sizes, labels=labels_with_percent, colors=colors,
                            autopct='%1.1f%%', shadow=True, startangle=90)
                    plt.axis('equal')
                    plt.title(f'{organ.capitalize()} Injury Severity Distribution', fontsize=15)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'injury_severity_distribution.png'), dpi=300)
        plt.show()

    # 检查是否有缺失值
    print("\nChecking for missing values:")
    missing_values = train_df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])

        # 可视化缺失值
        plt.figure(figsize=(10, 6))
        missing_cols = missing_values[missing_values > 0].index
        plt.bar(missing_cols, missing_values[missing_values > 0], color='crimson')
        plt.xlabel('Column')
        plt.ylabel('Missing Value Count')
        plt.title('Missing Values in Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values.png'), dpi=300)
        plt.show()
    else:
        print("No missing values in the dataset")

    # 创建一个器官状态的映射表，用于后续处理
    organ_status_map = {}
    for index, row in train_df.iterrows():
        patient_id = row['patient_id']
        organ_statuses = {}
        for organ in organs:
            # 检查该器官的所有可能状态列
            status_columns = [col for col in train_df.columns if col.startswith(f'{organ}_')]

            # 记录该器官的状态
            organ_status = {}
            for col in status_columns:
                status_type = col.split('_')[1]  # 获取状态类型（healthy, injury, low, high）
                organ_status[status_type] = row[col]

            organ_statuses[organ] = organ_status

        organ_status_map[patient_id] = organ_statuses

    # 构建 segmentation_map
    # 利用 train_series_meta 文件建立 series_id 到 patient_id 的映射
    train_series_meta = pd.read_csv(TRAIN_META)
    # 假设 train_series_meta 包含 'series_id' 和 'patient_id' 两列
    series_to_patient = dict(zip(train_series_meta['series_id'], train_series_meta['patient_id']))
    segmentation_map = {}
    for f in glob.glob(os.path.join(SEGMENTATION_DIR, "*.nii")):
        try:
            series_id = int(os.path.splitext(os.path.basename(f))[0])
            segmentation_map[str(series_id)] = f # force it to be string
        except ValueError:
            print(f"Could not extract series_id from {f}")

    # 保存图表
    print(f"\nVisualization results saved to {OUTPUT_DIR}")

    return train_df, organ_status_map, segmentation_map


# 加载并处理数据
train_df, organ_status_map, segmentation_map = load_and_process_data()
# 示例：打印部分 segmentation_map 信息
print("部分 segmentation_map 信息：")
for pid, seg_file in list(segmentation_map.items())[:5]:
    print(f"Patient ID: {pid} -> Segmentation file: {seg_file}")



def segment_slice(image, model, img_size=(224, 224)):
    """
    使用训练好的U-Net模型对单张切片进行分割
    
    Args:
        image: 单张CT切片图像 (numpy array)，形状应为 (height, width) 或 (height, width, channels)
               其中 channels 可以为 1 或 3
        model: 训练好的U-Net模型 (tensorflow.keras.Model)
        img_size: 图像大小（tuple），用于处理不同尺寸的图像

    Returns:
        pred_mask: 分割后的掩码 (numpy array)，形状为 (height, width)
    """

    # 1. 确保图像是正确的形状
    if len(image.shape) == 2:  # 如果是单通道的2D图像，添加通道维度
        image = np.expand_dims(image, axis=-1)
    elif len(image.shape) == 3 and image.shape[-1] > 3: # 如果通道数大于3 报错
        raise ValueError("Image has more than 3 channels.  It should have 1 or 3")
    
    # 2. 确保图像是3通道的
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # 3. 缩放到模型所需的尺寸
    if image.shape[0] != img_size[0] or image.shape[1] != img_size[1]:
        image = cv2.resize(image, img_size)  # 使用 cv2 缩放图像

    # 4. 转换为 float32
    image = image.astype(np.float32)

    # 5. 归一化像素值到 [0, 1] 范围 (如果需要)
    max_val = np.max(image)
    if max_val != 0:
        image = image / max_val

    # 6. 添加批次维度
    input_image = np.expand_dims(image, axis=0)

    # 7. 使用U-Net模型进行分割
    pred_mask = model.predict(input_image)[0]

    # 8. 从模型输出中提取单通道掩码
    pred_mask = pred_mask[:, :, 0]

    # 9. 确保输出掩码形状正确
    assert len(pred_mask.shape) == 2, f"Mask should be 2D, but got shape {pred_mask.shape}"
    return pred_mask


class UNetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map,
                 batch_size=8, img_size=(224, 224), shuffle=True, debug=False):
        self.patient_ids = patient_ids
        self.meta_df = meta_df
        self.dicom_tags_df = dicom_tags_df
        self.segmentation_map = segmentation_map
        self.organ_status_map = organ_status_map
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.debug = debug
        self.indexes = np.arange(len(self.patient_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.patient_cache = {}

    def __len__(self):
        return int(np.ceil(len(self.patient_ids) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.patient_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_patient_ids = [self.patient_ids[i] for i in indexes]
        return self._generate_unet_batch(batch_patient_ids)
    
    def _load_patient_data(self, patient_id):
        """This function is largely the same as the old function."""
        if patient_id in self.patient_cache:
            return self.patient_cache[patient_id]
        
        try:
            subset = self.meta_df[self.meta_df['patient_id'] == patient_id]
            series_id = str(subset['series_id'].iloc[0])
        except Exception as e:
            print(f"No series information found for patient {patient_id}. Error: {e}")
            return None

        segmentation_file = self.segmentation_map.get(series_id)
        
        has_segmentation = False
        segmentation_data = None
        if segmentation_file:
            try:
                segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
                has_segmentation = True
                resized_segmentation_data = resize(
                    segmentation_data,
                    self.img_size,
                    order=0,
                    preserve_range=True
                ).astype(np.float32)
            except Exception as e:
                print(f"Error loading the Segmentation for {series_id}: {e}")
                has_segmentation = False
        
        if self.dicom_tags_df is not None:
            image_ids = self.dicom_tags_df[
                (self.dicom_tags_df['PatientID'] == patient_id) &
                (self.dicom_tags_df['series_id'] == series_id)
            ]['InstanceNumber'].values
            image_ids = sorted(image_ids)
        else:
            print(f"No image_id information found for patient {patient_id} and series {series_id}.")
            return None

        if len(image_ids) == 0:
            print(f"No image IDs found for patient {patient_id} and series {series_id}!")
            return None

        patient_slices = []
        for image_id in image_ids:
            image_file = os.path.join(TRAIN_IMAGES, str(patient_id), str(series_id), f'{image_id}.dcm')
            try:
                dicom = pydicom.dcmread(image_file)
                image = dicom.pixel_array.astype(np.float32)
                image = cv2.resize(image, self.img_size)
                max_val = np.max(image)
                image = image / max_val if max_val != 0 else image.astype(np.float32)
                organ_statuses = self.organ_status_map.get(patient_id)
                
                mask = np.zeros(self.img_size, dtype=np.float32)
                if has_segmentation:
                    if image_id <= segmentation_data.shape[2]:
                        slice_mask = resize(
                            segmentation_data[:, :, image_id-1],
                            self.img_size,
                            order=0,
                            preserve_range=True
                        ).astype(np.float32)
                        mask = slice_mask
                slice_data = {
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_id': image_id,
                    'image': image,
                    'mask': mask,
                    'organ_statuses': organ_statuses,
                    'instance_number': dicom.InstanceNumber
                }
                patient_slices.append(slice_data)
                del dicom
                del image
            except Exception as e:
                print(f"Error reading DICOM file {image_file}: {e}")
                continue
        print(f"Patient {patient_id} loaded with {len(patient_slices)} slices")
        result = {
            'patient_id': patient_id,
            'series_id': series_id,
            'slices': patient_slices,
            'has_segmentation': has_segmentation,
            'segmentation_data': segmentation_data if has_segmentation else None
        }
        self.patient_cache[patient_id] = result
        return result

    def _generate_unet_batch(self, batch_patient_ids):
        batch_images = []
        batch_masks = []
        
        for patient_id in batch_patient_ids:
            print(f"Generating UNet data for patient {patient_id}")
            patient_data = self._load_patient_data(patient_id)
            if patient_data is None:
                print(f"Skipping patient {patient_id} due to missing data")
                continue
                    
            slices = patient_data['slices']
            if len(slices) == 0:
                print(f"Skipping patient {patient_id} due to missing slices")
                continue
            
            slice_data = random.choice(slices)
            image = slice_data['image']
            mask = slice_data['mask']
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
                    
            image, mask = augment_data(image, mask)

            batch_images.append(image)
            batch_masks.append(mask)
            del image, mask, slice_data, slices, patient_data
        
        if not batch_images:
            return np.empty((0, *self.img_size, 3)), np.empty((0, *self.img_size, 1))
                
        batch_images = np.array(batch_images)
        batch_masks = np.array(batch_masks)
        print(f"[U-Net] Generated batch: images shape={batch_images.shape}, masks shape={batch_masks.shape}")
        return batch_images, batch_masks



class DataGenerator(tf.keras.utils.Sequence):
    """数据生成器，用于分批次加载和处理CT图像数据，使用U-Net分割结果"""

    def __init__(self, patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map,
                 unet_model, batch_size=8, img_size=(224, 224), shuffle=True, debug=False):
        """
        初始化数据生成器

        参数:
            patient_ids: 患者ID列表
            meta_df: 包含series_id信息的元数据DataFrame
            dicom_tags_df: 包含DICOM标签的DataFrame
            segmentation_map: 分割文件映射字典
            organ_status_map: 器官状态映射字典
            unet_model: 训练好的U-Net模型 (用于分割)
            batch_size: 批次大小
            img_size: 图像大小
            shuffle: 是否打乱数据
            mode: 'classifier'用于2.5D分类模型
            debug: 是否为调试模式（此处为 False 时使用全部数据）
        """
        self.patient_ids = patient_ids
        self.meta_df = meta_df
        self.dicom_tags_df = dicom_tags_df
        self.segmentation_map = segmentation_map
        self.organ_status_map = organ_status_map
        self.unet_model = unet_model
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.mode = mode
        self.debug = debug
        self.organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        self.indexes = np.arange(len(self.patient_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.patient_cache = {}

    def __len__(self):
        return int(np.ceil(len(self.patient_ids) / self.batch_size))

    def __getitem__(self, index):
        print(f"正在生成第 {index+1} 个批次")
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_patient_ids = [self.patient_ids[i] for i in indexes]
        print(f"本批次患者ID: {batch_patient_ids}")

        return self._generate_classifier_batch(batch_patient_ids)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.patient_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_patient_data(self, patient_id):
        if patient_id in self.patient_cache:
            return self.patient_cache[patient_id]

        try:
            subset = self.meta_df[self.meta_df['patient_id'] == patient_id]
            series_id = str(subset['series_id'].iloc[0])
            print(f"患者 {patient_id} 对应的 series_id: {series_id}")
        except Exception as e:
            print(f"No series information found for patient {patient_id}. Error: {e}")
            return None

        segmentation_file = self.segmentation_map.get(series_id)

        has_segmentation = False
        segmentation_data = None
        if segmentation_file:
            try:
                segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
                has_segmentation = True
                resized_segmentation_data = resize(
                    segmentation_data,
                    self.img_size,
                    order=0,
                    preserve_range=True
                ).astype(np.float32)
                print(f"患者 {patient_id} 成功加载分割数据。")
            except Exception as e:
                print(f"Error loading the Segmentation for {series_id}: {e}")
                has_segmentation = False

        if self.dicom_tags_df is not None:
            image_ids = self.dicom_tags_df[(self.dicom_tags_df['PatientID'] == patient_id) & 
                                      (self.dicom_tags_df['series_id'] == series_id)]['InstanceNumber'].values
            image_ids = sorted(image_ids)
        else:
            print(f"No image_id information found for patient {patient_id} and series {series_id}.")
            return None

        if len(image_ids) == 0:
            print(f"No image IDs found for patient {patient_id} and series {series_id}!")
            return None

        print(f"患者 {patient_id} 序列 {series_id} 共找到 {len(image_ids)} 张切片")

        patient_slices = []
        for image_id in image_ids:
            image_file = os.path.join(TRAIN_IMAGES, str(patient_id), str(series_id), f'{image_id}.dcm')
            try:
                dicom = pydicom.dcmread(image_file)
                image = dicom.pixel_array.astype(np.float32)
                image = cv2.resize(image, self.img_size)
                max_val = np.max(image)
                image = image / max_val if max_val != 0 else image.astype(np.float32)
                organ_statuses = self.organ_status_map.get(patient_id)
                
                mask = np.zeros(self.img_size, dtype=np.float32)
                if has_segmentation:
                    if image_id <= segmentation_data.shape[2]:
                        slice_mask = resize(
                            segmentation_data[:, :, image_id-1],
                            self.img_size,
                            order=0,
                            preserve_range=True
                        ).astype(np.float32)
                        mask = slice_mask
                
                slice_data = {
                    'patient_id': patient_id,
                    'series_id': series_id,
                    'image_id': image_id,
                    'image': image,
                    'mask': mask,
                    'organ_statuses': organ_statuses,
                    'instance_number': dicom.InstanceNumber
                }
                patient_slices.append(slice_data)
                del dicom
                del image
            except Exception as e:
                print(f"Error reading DICOM file {image_file}: {e}")
                continue
        print(f"患者 {patient_id} 加载完成，切片数量: {len(patient_slices)} ")
        result = {
            'patient_id': patient_id,
            'series_id': series_id,
            'slices': patient_slices,
            'has_segmentation': has_segmentation,
            'segmentation_data': segmentation_data if has_segmentation else None
        }
        self.patient_cache[patient_id] = result
        del segmentation_data,resized_segmentation_data
        gc.collect()
        return result

    def _generate_classifier_batch(self, batch_patient_ids):
        batch_sequences = []
        batch_labels = []

        for patient_id in batch_patient_ids:
            print(f"Processing patient {patient_id} for classifier batch")
            patient_data = self._load_patient_data(patient_id)
            if patient_data is None:
                print(f"Skipping patient {patient_id} due to missing data")
                continue

            slices = patient_data['slices']
            if len(slices) < 32:
                print(f"Skipping patient {patient_id} because they have less than 32 slices")
                continue
                # 确保列表长度大于等于32
            if len(slices) < 32:
                 print(f"患者 {patient_id} 的切片数量不足 32 张，跳过")
                 continue

            # 选择中间的32张连续切片
            middle_index = len(slices) // 2
            start_index = max(0, middle_index - 16)
            sequence = slices[start_index:start_index + 32]

            if len(sequence) < 32:
                print(f"提取连续32张切片失败，跳过患者{patient_id}")
                continue

            processed_sequence = []
            print("正在处理batch")
            for slice_data in sequence:
                image = slice_data['image']
                #将分割后的图像放进去, 并且使用unet分割后的图像，
                #TODO 添加分割的代码，将unet_model作为参数传递进来后，实现分割功能。
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)
                
                segmented_slice = segment_slice(image, self.unet_model, self.img_size) #call the segmentation process
                processed_sequence.append(segmented_slice) #append the new segmented slice.

                del image, segmented_slice
                gc.collect()

            print("处理完毕")
            organ_status = self.organ_status_map.get(patient_id)
            if not organ_status:
                print(f"不存在{patient_id}的数据集")
                continue

            labels = []
            for organ in self.organs:
                status = organ_status.get(organ, {})
                organ_label = [status.get('healthy', 0),
                               status.get('injury', 0),
                               status.get('low', 0),
                               status.get('high', 0)]
                labels.append(organ_label)

            processed_sequence_arr = np.array(processed_sequence, dtype=np.float32) #make sure this is the proper format before adding to sequence. 
            batch_sequences.append(processed_sequence_arr)
            batch_labels.append(np.array(labels, dtype=np.float32))

            del processed_sequence, labels, slice_data, slices, patient_data, sequence,processed_sequence_arr
            gc.collect()

        if not batch_sequences:
            print("当前批次为空，返回空数组")
            return np.empty((0, 32, *self.img_size, 3),dtype=np.float32), np.empty((0, len(self.organs), 4),dtype=np.float32)
        
        batch_sequences = np.array(batch_sequences,dtype=np.float32)
        batch_labels = np.array(batch_labels,dtype=np.float32)
        print(f"2.5D分类 成功生成批次，序列shape：{batch_sequences.shape}，标签shape：{batch_labels.shape}")
        gc.collect()
        return batch_sequences, batch_labels



def augment_data(image, mask=None):
    """
    对图像和掩码（可选）应用数据增强。
    """
    # 确保图像形状正确
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]  # 如果是单通道的3D图像，转为2D
    
    original_shape = image.shape
    
    # 随机水平翻转
    if random.random() > 0.5:
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
    
    # 随机垂直翻转
    if random.random() > 0.5:
        image = np.flipud(image)
        if mask is not None:
            mask = np.flipud(mask)
    
    # 随机旋转
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE[0]/2, IMG_SIZE[1]/2), angle, 1)
    image = cv2.warpAffine(image, M, IMG_SIZE)
    if mask is not None:
        mask = cv2.warpAffine(mask, M, IMG_SIZE, flags=cv2.INTER_NEAREST)
    
    # 模糊
    if random.random() > 0.5:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 高斯噪声
    if random.random() > 0.5:
        if len(original_shape) == 2:  # 2D图像
            row, col = image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col))
            image = image + gauss
        elif len(original_shape) == 3:  # 3D图像
            row, col, ch = image.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            image = image + gauss
    
    # 确保值范围在[0,1]
    image = np.clip(image, 0, 1)
    
    if mask is not None:
        return image, mask
    else:
        return image



def build_unet(input_shape):
    """
    构建基于EfficientNetB0的2D U-Net模型，用于单张CT切片分割。
    论文中使用的架构。
    input_shape应为(height, width, channels)
    """
    # 加载预训练的EfficientNetB0作为编码器
    efficientnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # 获取编码器的中间层特征图
    c1 = efficientnet.get_layer('block2b_add').output  # 64x64
    c2 = efficientnet.get_layer('block3b_add').output  # 32x32
    c3 = efficientnet.get_layer('block5c_add').output  # 16x16
    c4 = efficientnet.get_layer('block6d_add').output  # 8x8
    b0 = efficientnet.output  # 瓶颈: 8x8
    
    # 解码器部分 - 对称上采样
    # 步骤1: 8x8 -> 16x16
    up1 = layers.Conv2DTranspose(512, (3,3), strides=2, padding='same', activation='relu')(b0)
    merge1 = layers.concatenate([c3, up1], axis=3)
    conv1 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge1)
    conv1 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv1)
    
    # 步骤2: 16x16 -> 32x32
    up2 = layers.Conv2DTranspose(256, (3,3), strides=2, padding='same', activation='relu')(conv1)
    merge2 = layers.concatenate([c2, up2], axis=3)
    conv2 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge2)
    conv2 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv2)
    
    # 步骤3: 32x32 -> 64x64
    up3 = layers.Conv2DTranspose(128, (3,3), strides=2, padding='same', activation='relu')(conv2)
    merge3 = layers.concatenate([c1, up3], axis=3)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge3)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    
    # 步骤4: 64x64 -> 128x128
    up4 = layers.Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')(conv3)
    
    # 步骤5: 128x128 -> 256x256 (如果需要)
    up5 = layers.Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')(up4)
    
    # 处理尺寸不匹配问题
    # 检查输出尺寸是否需要裁剪
    if up5.shape[1] != input_shape[0] or up5.shape[2] != input_shape[1]:
        # 计算需要裁剪的边缘大小
        crop_height = int((up5.shape[1] - input_shape[0]) / 2) if up5.shape[1] > input_shape[0] else 0
        crop_width = int((up5.shape[2] - input_shape[1]) / 2) if up5.shape[2] > input_shape[1] else 0
        
        if crop_height > 0 or crop_width > 0:
            up5 = layers.Cropping2D(cropping=((crop_height, crop_height), 
                                             (crop_width, crop_width)))(up5)
    
    # 输出层 - 单通道分割掩码
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(up5)
    
    # 创建模型
    model = models.Model(inputs=efficientnet.input, outputs=outputs)
    
    return model



def build_25d_classifier(input_shape, num_organs=5):
    """
    构建2.5D分类模型，使用32张连续切片作为输入。
    input_shape应为单张图像的形状(height, width, channels)
    """
    # 输入层 - 32张连续切片
    input_ct = layers.Input(shape=(32,) + input_shape, name='ct_input')
    
    # 使用TimeDistributed包装EfficientNetB1，处理每张切片
    # 加载预训练的EfficientNetB1，但移除顶层
    base_model = EfficientNetB1(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False  # 冻结基础模型权重
    
    # 使用TimeDistributed应用到每张切片
    x = layers.TimeDistributed(base_model)(input_ct)
    
    # 双向LSTM提取时序特征
    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=False))(x)
    
    # 全连接层
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # 输出层 - 每个器官4个类别(healthy, injury, low, high)
    output = layers.Dense(num_organs * 4, activation='sigmoid')(x)
    output = layers.Reshape((num_organs, 4))(output)
    
    model = models.Model(inputs=input_ct, outputs=output)
    return model



def weighted_cross_entropy(organ_weights):
    """
    Returns a weighted cross-entropy loss function.
    """
    def loss(y_true, y_pred):
        """
        Calculates the weighted cross-entropy loss.
        """
        loss = 0.0
        for i in range(NUM_ORGANS):
            # 提取每个器官的真实标签和预测值
            y_true_organ = y_true[:, i, :]
            y_pred_organ = y_pred[:, i, :]

            # 计算交叉熵损失
            cross_entropy = tf.keras.losses.binary_crossentropy(y_true_organ, y_pred_organ)

            # 应用权重
            loss += organ_weights[i] * tf.reduce_mean(cross_entropy)

        return loss
    return loss

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculates evaluation metrics."""
    y_pred_binary = (y_pred > threshold).astype(int)  # 将预测概率转换为二进制标签
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    if len(np.unique(y_true)) > 1:  # 避免二分类混淆矩阵只有一类时出错
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0  # 如果只有一类，则特异度无法计算
    ppv = precision
    npv = 0 # 默认 npv 为 0 ， 在计算出 cm 后更新数值
    if len(np.unique(y_true)) > 1:
      npv = tn / (tn + fn) if (tn + fn) > 0 else 0 # 根据 tn 计算 npv
    return accuracy, precision, recall, specificity, ppv, npv, confusion_matrix(y_true, y_pred_binary)



def main():
    # 1. 加载和预处理数据、器官状态和分割映射
    train_df, organ_status_map, segmentation_map = load_and_process_data()
    
    # 2. 划分患者ID为训练集和验证集
    from sklearn.model_selection import train_test_split
    patient_ids = train_df['patient_id'].unique()
    train_patient_ids, val_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    
    # 3. 加载元数据和DICOM标签
    meta_df = pd.read_csv(TRAIN_META)
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    dicom_tags_df = pd.read_parquet(TRAIN_DICOM_TAGS)
    temp = dicom_tags_df['SeriesInstanceUID'].str.split('.', expand=True)
    dicom_tags_df['series_id'] = temp[8]
    dicom_tags_df['series_id'] = dicom_tags_df['series_id'].astype(str)
    dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
    

    # 4. 构建U-Net模型
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    unet_model = build_unet(input_shape)
    unet_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    unet_model.summary()

    #create the data Gen and now train the U-net on It.
    unet_train_generator = UNetDataGenerator(
        train_patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map,
        batch_size=2, img_size=IMG_SIZE, shuffle=True, # The UNet dataloader doesn't require UNET,
    )
    unet_val_generator = UNetDataGenerator(
        val_patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map,
        batch_size=2, img_size=IMG_SIZE, shuffle=False,
    )
    gc.collect() # Explicitly run garbage collection

    history_unet = unet_model.fit(
        unet_train_generator,
        validation_data=unet_val_generator,
        epochs=5, # Reduce for testing
    )
    unet_model.save('unet_model.h5') #Save the UNet after training the UNet
    #deallocate memory, and remove them so that it reloads and reduces future memory use.
    del train_patient_ids, val_patient_ids,unet_train_generator, unet_val_generator
    gc.collect() # Run garbage collection

    # 5. 重新加载需要的数据 (需要重新加载，因为之前删除了)
    train_df, organ_status_map, segmentation_map = load_and_process_data()
    meta_df = pd.read_csv(TRAIN_META)
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    dicom_tags_df = pd.read_parquet(TRAIN_DICOM_TAGS)
    temp = dicom_tags_df['SeriesInstanceUID'].str.split('.', expand=True)
    dicom_tags_df['series_id'] = temp[8]
    dicom_tags_df['series_id'] = dicom_tags_df['series_id'].astype(str)
    dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
    
    patient_ids = train_df['patient_id'].unique()
    train_patient_ids, val_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)


    # 6: Building 2.5D Classifier
    #Load Generator
    train_generator_classifier = DataGenerator(
        train_patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map, unet_model, #Pass unet to data generator to load the unet model
        batch_size=2, img_size=IMG_SIZE, mode='classifier', debug=False
    )
    val_generator_classifier = DataGenerator(
        val_patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map, unet_model,#Pass the unet model to the dataloader
        batch_size=2, img_size=IMG_SIZE, mode='classifier', debug=False, shuffle=False
    )

    gc.collect()
    del train_patient_ids, val_patient_ids, meta_df, dicom_tags_df, temp, train_df, organ_status_map, segmentation_map, patient_ids
    gc.collect() # Explicitly run garbage collection
    # 7. 构建 2.5D 分类模型
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    classifier_model = build_25d_classifier(input_shape=input_shape, num_organs=NUM_ORGANS)
    organ_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_function = weighted_cross_entropy(organ_weights)
    optimizer_inst = optimizers.Adam(learning_rate=1e-4)
    classifier_model.compile(optimizer=optimizer_inst, loss=loss_function, metrics=['accuracy'])
    classifier_model.summary()

    # 8. 使用分割结果训练2.5D分类模型
    history_classifier = classifier_model.fit(
        train_generator_classifier,
        validation_data=val_generator_classifier,
        epochs=5,
    )
    classifier_model.save('classifier_model.h5')

    del train_generator_classifier, val_generator_classifier #deallocate training generators
    gc.collect() #Run garbage collection

    #deallocate memory for UNet Model
    del unet_model
    gc.collect()

    # 9. 评估 2.5D 分类模型
    print("现在开始评估分类器")
    #ReLoading data sets again
    train_df, organ_status_map, segmentation_map = load_and_process_data()
    meta_df = pd.read_csv(TRAIN_META)
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    dicom_tags_df = pd.read_parquet(TRAIN_DICOM_TAGS)
    temp = dicom_tags_df['SeriesInstanceUID'].str.split('.', expand=True)
    dicom_tags_df['series_id'] = temp[8]
    dicom_tags_df['series_id'] = dicom_tags_df['series_id'].astype(str)
    dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
    patient_ids = train_df['patient_id'].unique()
    
    #Load the validation data Generator
    val_generator_classifier = DataGenerator(
        patient_ids, meta_df, dicom_tags_df, segmentation_map, organ_status_map,
        unet_model, batch_size=2, img_size=IMG_SIZE, mode='classifier', debug=False, shuffle=False #Do not shuffle the eval set
    )
    
        #deallocate memory we've used.
    del train_df, meta_df, dicom_tags_df, temp #deallocate training generators
    gc.collect() # Explicitly run garbage collection

    val_ground_truth = []
    val_predictions = []
    for i in range(len(val_generator_classifier)):
        X_val, y_val = val_generator_classifier[i]

        #验证是否有batch
        if X_val.shape[0] > 0:
            y_pred = classifier_model.predict(X_val)
            val_ground_truth.extend(y_val)
            val_predictions.extend(y_pred)
    
    del X_val, y_val, val_generator_classifier #Explicit deallocation
    gc.collect() #Garbage Collection
    
    val_ground_truth = np.array(val_ground_truth)
    val_predictions = np.array(val_predictions)
    gc.collect()#deallocate inside eval loop
    
    if len(val_predictions) > 0:
        print("完成验证集评估")
        organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
        for i, organ in enumerate(organs):
            print(f"器官: {organ}")
            y_true_organ = val_ground_truth[:, i, 1]
            y_pred_organ = val_predictions[:, i, 1]
            accuracy, precision, recall, specificity, ppv, npv, cm = calculate_metrics(y_true_organ, y_pred_organ)
            print(f"  准确率: {accuracy:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  召回率: {recall:.4f}")
            print(f"  特异性: {specificity:.4f}")
            print(f"  PPV: {ppv:.4f}")
            print(f"  NPV: {npv:.4f}")
            print(f"  混淆矩阵: \n{cm}")
    
    del val_predictions, val_ground_truth ,classifier_model,  loss_function, optimizer_inst#Final deallocation
    gc.collect() #Garbage collection
gc.collect()#garbage collection at the very beginning.
if __name__ == "__main__":
    gc.collect()
    main()
    gc.collect()#garbage collection at the very end.







