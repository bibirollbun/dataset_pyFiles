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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import matplotlib as mpl
import matplotlib.font_manager as fm


# 检查GPU是否可用
print("TensorFlow版本:", tf.__version__)
print("GPU是否可用:", tf.config.list_physical_devices('GPU'))

# 限制TensorFlow最多占用显存而非一把吃掉
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)


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
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
SEGMENTATION_DIR = "/kaggle/input/rsna-2023-abdominal-trauma-detection/segmentations"
TRAIN_DICOM_TAGS = '/kaggle/input/rsna-2023-abdominal-trauma-detection/train_dicom_tags.parquet' 

# 图像尺寸统一为论文中的224×224
IMG_SIZE = (224, 224)
NUM_ORGANS = 5  # 肝脏, 脾脏, 肾脏, 肠道, 外渗

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)



def load_and_process_data():
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df['patient_id'] = train_df['patient_id'].astype(str)
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    organ_status_map = {}
    for _, row in train_df.iterrows():
        pid = row['patient_id']
        statuses = {}
        for organ in organs:
            status_columns = [col for col in train_df.columns if col.startswith(f'{organ}_')]
            organ_status = {col.split('_')[1]: row[col] for col in status_columns}
            statuses[organ] = organ_status
        organ_status_map[pid] = statuses

    train_series_meta = pd.read_csv(TRAIN_META)
    series_to_patient = dict(zip(
        train_series_meta['series_id'].astype(str),
        train_series_meta['patient_id'].astype(str)
    ))

    segmentation_map = {}
    for fpath in glob.glob(os.path.join(SEGMENTATION_DIR, "*.nii")):
        name = os.path.splitext(os.path.basename(fpath))[0]
        if name in series_to_patient:
            segmentation_map[series_to_patient[name]] = fpath

    del train_series_meta, series_to_patient
    gc.collect()

    print(f"Preprocessing complete. Processed {len(train_df)} records.")
    return train_df, organ_status_map, segmentation_map


def visualize_data(train_df, organ_status_map=None):
    """Visualize organ status distributions and missing values"""
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']

    # 1. Organ status pie charts
    plt.figure(figsize=(20, 15))
    for i, organ in enumerate(organs):
        plt.subplot(2, 3, i+1)
        status_data = {}
        total = len(train_df)
        for label in ['healthy', 'injury', 'low', 'high']:
            col = f"{organ}_{label}"
            if col in train_df.columns:
                count = train_df[col].sum()
                status_data[label.capitalize() if label == 'healthy' else ('Low-grade Injury' if label=='low' else 'High-grade Injury' if label=='high' else 'Injury')] = count
        if not status_data:
            continue
        sizes = list(status_data.values())
        labels = list(status_data.keys())
        sizes_pct = [s/total*100 for s in sizes]
        labels_pct = [f"{l}: {p:.1f}%" for l, p in zip(labels, sizes_pct)]
        colors = plt.cm.Paired(np.arange(len(sizes)) / len(sizes))
        explode = [0.1 if 'Injury' in l else 0 for l in labels]
        plt.pie(sizes, explode=explode, labels=labels_pct, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(f"{organ.capitalize()} Status Distribution", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'organ_status_distribution.png'), dpi=300)
    plt.show()

    # 2. Injury percentage bar chart
    percentages = []
    for organ in organs:
        inj = 0
        for label in ['injury', 'low', 'high']:
            col = f"{organ}_{label}"
            if col in train_df.columns:
                inj += train_df[col].sum()
        percentages.append(inj/len(train_df)*100)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(organs, percentages, color=plt.cm.viridis(np.linspace(0, 1, len(organs))))
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{pct:.1f}%", ha='center', va='bottom', fontsize=12)
    plt.xlabel('Organ', fontsize=14)
    plt.ylabel('Injury Percentage (%)', fontsize=14)
    plt.title('Injury Percentage by Organ', fontsize=16)
    plt.ylim(0, max(percentages)*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, 'organ_injury_percentages.png'), dpi=300)
    plt.show()

    # 3. Severity distribution pie charts
    severity_organs = [o for o in organs if any(f"{o}_{lab}" in train_df.columns for lab in ['low','high'])]
    if severity_organs:
        plt.figure(figsize=(14, 10))
        for i, organ in enumerate(severity_organs):
            plt.subplot(2, 3, i+1)
            sizes = []
            labels = []
            for lab, name in [('low', 'Low-grade Injury'), ('high', 'High-grade Injury')]:
                col = f"{organ}_{lab}"
                if col in train_df.columns:
                    cnt = train_df[col].sum()
                    sizes.append(cnt)
                    labels.append(f"{name}: {cnt/ (sum(sizes)) * 100:.1f}%")
            if not sizes or sum(sizes)==0:
                continue
            plt.pie(sizes, labels=labels, colors=['#ff9999','#ff3333'],
                    autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')
            plt.title(f"{organ.capitalize()} Injury Severity Distribution", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'injury_severity_distribution.png'), dpi=300)
        plt.show()

    # 4. Missing values
    missing = train_df.isnull().sum()
    if missing.sum() > 0:
        plt.figure(figsize=(10, 6))
        cols = missing[missing>0].index
        vals = missing[missing>0].values
        plt.bar(cols, vals, color='crimson')
        plt.xlabel('Column')
        plt.ylabel('Missing Value Count')
        plt.title('Missing Values in Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values.png'), dpi=300)
        plt.show()
    else:
        print("No missing values in the dataset")


# 加载并处理数据
train_df, organ_status_map, segmentation_map = load_and_process_data()
gc.collect()
visualize_data(train_df)
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
    try:
        # 1. 确保图像是正确的形状
        if len(image.shape) == 2:  # 如果是单通道的2D图像，添加通道维度
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and image.shape[-1] > 3: # 如果通道数大于3 报错
            raise ValueError("Image has more than 3 channels. It should have 1 or 3")
        
        # 打印图像统计信息
        print(f"Input image stats: shape={image.shape}, max={np.max(image):.4f}, min={np.min(image):.4f}, mean={np.mean(image):.4f}")
        
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
        pred_mask = model.predict(input_image, verbose=0)[0]
        
        # 打印预测掩码统计信息
        print(f"Pred mask stats: shape={pred_mask.shape}, max={np.max(pred_mask):.4f}, min={np.min(pred_mask):.4f}, mean={np.mean(pred_mask):.4f}")

        # 8. 从模型输出中提取单通道掩码
        pred_mask = pred_mask[:, :, 0]

        # 9. 确保输出掩码形状正确
        assert len(pred_mask.shape) == 2, f"Mask should be 2D, but got shape {pred_mask.shape}"
        return pred_mask
        
    except Exception as e:
        print(f"Error in segment_slice: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # 返回一个空掩码
        return np.zeros(img_size, dtype=np.float32)


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
    # 检查并修正输入通道数
    if input_shape[-1] == 1:
        # 如果是单通道，修改为3通道以适应EfficientNet要求
        efficientnet_input_shape = (input_shape[0], input_shape[1], 3)
        print(f"注意: 将输入形状从 {input_shape} 修改为 {efficientnet_input_shape} 以适应EfficientNet")
    else:
        efficientnet_input_shape = input_shape
    
    # 输入层 - 32张连续切片
    input_ct = layers.Input(shape=(32,) + input_shape, name='ct_input')
    
    # 如果是单通道输入，需要转换为三通道
    if input_shape[-1] == 1:
        # 将单通道转换为三通道
        x = layers.TimeDistributed(layers.Conv2D(3, kernel_size=1, padding='same'))(input_ct)
    else:
        x = input_ct
    
    # 使用TimeDistributed包装EfficientNetB1，处理每张切片
    # 加载预训练的EfficientNetB1，但移除顶层
    base_model = EfficientNetB1(
        include_top=False, 
        weights='imagenet', 
        input_shape=efficientnet_input_shape,  # 使用修正后的形状
        pooling='avg'
    )
    base_model.trainable = False  # 冻结基础模型权重
    
    # 使用TimeDistributed应用到每张切片
    x = layers.TimeDistributed(base_model)(x)
    
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



def load_single_slice(args):
    """
    加载单个切片的函数，用于并行处理
    
    Args:
        args: 包含image_file, image_id, img_size, segmentation_data的元组
        
    Returns:
        切片数据字典或None（如果处理失败）
    """
    image_file, image_id, img_size, segmentation_data = args
    
    try:
        dicom = pydicom.dcmread(image_file)
        image = dicom.pixel_array.astype(np.float32)
        
        # 打印图像统计信息
        print(f"Image (ID: {image_id}) max: {np.max(image):.4f}, min: {np.min(image):.4f}, mean: {np.mean(image):.4f}")
        
        image = cv2.resize(image, img_size)
        max_val = np.max(image)
        image = image / max_val if max_val != 0 else image
        
        mask = np.zeros(img_size, dtype=np.float32)
        if segmentation_data is not None and image_id <= segmentation_data.shape[2]:
            mask_slice = segmentation_data[:, :, image_id-1]
            mask = cv2.resize(mask_slice, img_size, interpolation=cv2.INTER_NEAREST)
            
            # 打印掩码统计信息
            print(f"Mask (ID: {image_id}) max: {np.max(mask):.4f}, min: {np.min(mask):.4f}, mean: {np.mean(mask):.4f}")
            
        return {
            'image': image,
            'mask': mask,
            'instance_number': int(image_id)
        }
    except Exception as e:
        print(f"Error loading slice {image_id}: {str(e)}")
        return None


# 添加缓存功能 - 预处理和保存患者数据
def preprocess_patient_data(patient_id, meta_df, dicom_tags_df, segmentation_map, 
                           unet_model, output_dir, img_size=(224, 224), max_slices=32):
    """
    预处理单个患者的数据并保存到磁盘
    
    Args:
        patient_id: 患者ID
        meta_df: 元数据DataFrame
        dicom_tags_df: DICOM标签DataFrame
        segmentation_map: 分割映射
        unet_model: U-Net模型用于分割
        output_dir: 输出目录
        img_size: 图像大小
        max_slices: 每个患者保存的最大切片数（保持32张）
    
    Returns:
        成功处理返回True，否则返回False
    """
    output_file = os.path.join(output_dir, f"{patient_id}.npz")
    if os.path.exists(output_file):
        return True
        
    try:
        # 获取患者的系列ID
        subset = meta_df[meta_df['patient_id'] == patient_id]
        if subset.empty:
            return False
        series_id = str(subset['series_id'].iloc[0])
        
        # 获取图像ID
        image_ids = dicom_tags_df[
            (dicom_tags_df['PatientID'] == patient_id) &
            (dicom_tags_df['series_id'] == series_id)
        ]['InstanceNumber'].values
        
        if len(image_ids) == 0:
            return False
            
        image_ids = sorted(image_ids)
        
        # 获取分割数据
        segmentation_file = segmentation_map.get(patient_id)
        segmentation_data = None
        if segmentation_file and os.path.exists(segmentation_file):
            try:
                segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
            except Exception:
                pass
        
        # 准备并行加载的参数
        image_files = [os.path.join(TRAIN_IMAGES, str(patient_id), str(series_id), f'{image_id}.dcm') 
                      for image_id in image_ids]
        
        # 使用并行处理加载切片
        args_list = [(image_file, image_id, img_size, segmentation_data) 
                    for image_file, image_id in zip(image_files, image_ids)]
        
        all_slices = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(load_single_slice, args_list))
            all_slices = [r for r in results if r is not None]
        
        # 如果没有切片，跳过
        if not all_slices:
            return False
        
        # 选择中间的max_slices张切片
        if len(all_slices) > max_slices:
            middle_index = len(all_slices) // 2
            start_index = max(0, middle_index - max_slices // 2)
            all_slices = all_slices[start_index:start_index + max_slices]
        
        # 如果切片数量不足，则跳过
        if len(all_slices) < max_slices:
            return False
        
        # 对每个切片应用分割
        segmented_slices = []
        for slice_data in all_slices:
            image = slice_data['image']
            if len(image.shape) == 2:
                image_for_seg = np.stack([image, image, image], axis=-1)
            else:
                image_for_seg = image
            
            segmented_slice = segment_slice(image_for_seg, unet_model, img_size)
            segmented_slices.append(np.expand_dims(segmented_slice, axis=-1))  # 添加通道维度
        
        segmented_images = np.array(segmented_slices)
        
        # 保存到磁盘
        np.savez_compressed(output_file, segmented_images=segmented_images)
        
        return True
        
    except Exception as e:
        print(f"处理患者 {patient_id} 时出错: {str(e)}")
        return False



def preprocess_patients_parallel(patient_ids, meta_df, dicom_tags_df, segmentation_map, 
                                organ_status_map, unet_model, output_dir, 
                                img_size=(224, 224), max_slices=32, batch_size=10):
    """
    并行预处理多个患者的数据
    
    Args:
        patient_ids: 患者ID列表
        meta_df: 元数据DataFrame
        dicom_tags_df: DICOM标签DataFrame
        segmentation_map: 分割映射
        organ_status_map: 器官状态映射
        unet_model: U-Net模型用于分割
        output_dir: 输出目录
        img_size: 图像大小
        max_slices: 每个患者保存的最大切片数（保持32张）
        batch_size: 每批处理的患者数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始为 {len(patient_ids)} 个患者处理标签数据...")
    
    # 先为每个患者保存标签数据
    for patient_id in tqdm(patient_ids, desc="处理患者标签"):
        labels_file = os.path.join(output_dir, f"{patient_id}_labels.npz")
        if not os.path.exists(labels_file):
            try:
                # 获取器官状态标签
                organ_status = organ_status_map.get(patient_id, {})
                labels = []
                organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
                for organ in organs:
                    status = organ_status.get(organ, {})
                    organ_label = [
                        status.get('healthy', 0),
                        status.get('injury', 0),
                        status.get('low', 0),
                        status.get('high', 0)
                    ]
                    labels.append(organ_label)
                
                labels = np.array(labels, dtype=np.float32)
                
                # 保存标签到磁盘
                np.savez_compressed(labels_file, labels=labels)
            except Exception as e:
                print(f"处理患者 {patient_id} 标签时出错: {str(e)}")
    
    print("标签数据处理完成，开始处理图像数据...")
    
    # 分批处理患者数据
    total_batches = (len(patient_ids) + batch_size - 1) // batch_size
    processed_count = 0
    skipped_count = 0
    
    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{total_batches}，共 {len(batch)} 个患者")
        
        # 使用多线程并行处理每个患者
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for patient_id in batch:
                # 检查是否已经处理过
                output_file = os.path.join(output_dir, f"{patient_id}.npz")
                if os.path.exists(output_file):
                    skipped_count += 1
                    continue
                
                future = executor.submit(
                    preprocess_patient_data,
                    patient_id, meta_df, dicom_tags_df, segmentation_map,
                    unet_model, output_dir, img_size, max_slices
                )
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                if future.result():
                    processed_count += 1
        
        # 每批处理完成后强制垃圾回收
        gc.collect()
        
        # 打印进度
        print(f"批次 {i//batch_size + 1}/{total_batches} 完成，总计处理: {processed_count}，跳过: {skipped_count}")
    
    print(f"预处理完成! 成功处理: {processed_count}，跳过: {skipped_count}")



def load_patient_data_for_tf(patient_id, cache_dir):
    """加载单个患者的数据，用于tf.data API"""
    try:
        # 加载缓存的图像数据
        image_file = os.path.join(cache_dir, f"{patient_id}.npz")
        image_data = np.load(image_file, allow_pickle=True)
        segmented_images = image_data['segmented_images']
        
        # 加载缓存的标签数据
        label_file = os.path.join(cache_dir, f"{patient_id}_labels.npz")
        label_data = np.load(label_file, allow_pickle=True)
        labels = label_data['labels']
        
        # 确保切片数量是32
        if segmented_images.shape[0] != 32:
            if segmented_images.shape[0] > 32:
                middle = segmented_images.shape[0] // 2
                start = middle - 16
                segmented_images = segmented_images[start:start+32]
            else:
                # 返回空数据，后续会被过滤掉
                return np.zeros((32, 224, 224, 1), dtype=np.float32), np.zeros((5, 4), dtype=np.float32)
        
        return segmented_images.astype(np.float32), labels.astype(np.float32)
    except Exception as e:
        print(f"加载患者 {patient_id} 的缓存数据时出错: {str(e)}")
        # 返回空数据，后续会被过滤掉
        return np.zeros((32, 224, 224, 1), dtype=np.float32), np.zeros((5, 4), dtype=np.float32)

def create_organ_balanced_dataset(patient_ids, cache_dir, batch_size=4, shuffle=True):
    """创建按器官平衡的数据集，确保每个批次包含各种器官的损伤样本"""
    print("创建按器官平衡的数据集...")
    
    # 按器官损伤情况分类患者
    organ_patients = {
        'bowel': {'injured': [], 'healthy': []},
        'extravasation': {'injured': [], 'healthy': []},
        'kidney': {'injured': [], 'healthy': []},
        'liver': {'injured': [], 'healthy': []},
        'spleen': {'injured': [], 'healthy': []},
    }
    
    # 分类每个患者
    for pid in tqdm(patient_ids, desc="按器官分类患者"):
        label_file = os.path.join(cache_dir, f"{pid}_labels.npz")
        if os.path.exists(label_file):
            try:
                labels = np.load(label_file)['labels']
                
                # 检查每个器官是否有损伤
                for i, organ in enumerate(['bowel', 'extravasation', 'kidney', 'liver', 'spleen']):
                    # 如果有损伤标签(injury, low, high中任一个)
                    if np.any(labels[i, 1:] > 0):
                        organ_patients[organ]['injured'].append(pid)
                    else:
                        organ_patients[organ]['healthy'].append(pid)
                        
            except Exception as e:
                print(f"处理患者 {pid} 标签时出错: {str(e)}")
                continue
    
    # 打印每类患者数量
    for organ, status_dict in organ_patients.items():
        print(f"{organ}: {len(status_dict['injured'])} 损伤患者, {len(status_dict['healthy'])} 健康患者")
    
    # 计算每个批次要包含的每种类型的样本数
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
    def data_generator():
        while True:
            batch_patients = []
            
            # 确保每个批次至少包含每个器官的1个损伤样本
            for organ in organs:
                if organ_patients[organ]['injured']:
                    # 随机选择该器官的1个损伤患者
                    injured_pid = random.choice(organ_patients[organ]['injured'])
                    if injured_pid not in batch_patients:
                        batch_patients.append(injured_pid)
            
            # 填充剩余位置，确保批次大小正确
            remaining_slots = batch_size - len(batch_patients)
            if remaining_slots > 0:
                # 创建所有健康患者的列表
                all_healthy = []
                for organ in organs:
                    all_healthy.extend([p for p in organ_patients[organ]['healthy'] if p not in batch_patients])
                
                # 去重
                all_healthy = list(set(all_healthy))
                
                if all_healthy:
                    # 随机选择健康患者填充批次
                    selected_healthy = random.sample(all_healthy, min(remaining_slots, len(all_healthy)))
                    batch_patients.extend(selected_healthy)
            
            # 如果批次大小不足，随机复制已有患者
            while len(batch_patients) < batch_size:
                batch_patients.append(random.choice(batch_patients))
            
            # 如果批次太大，随机删减
            if len(batch_patients) > batch_size:
                batch_patients = random.sample(batch_patients, batch_size)
            
            # 打乱批次中的患者顺序
            random.shuffle(batch_patients)
            
            # 加载数据
            batch_images = []
            batch_labels = []
            
            for pid in batch_patients:
                try:
                    images, labels = load_patient_data_for_tf(pid, cache_dir)
                    batch_images.append(images)
                    batch_labels.append(labels)
                except Exception as e:
                    print(f"加载患者 {pid} 数据时出错: {str(e)}")
                    continue
            
            if batch_images:  # 确保至少有一个有效样本
                yield np.array(batch_images), np.array(batch_labels)
    
    # 创建tf.data.Dataset
    output_signature = (
        tf.TensorSpec(shape=(None, 32, 224, 224, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5, 4), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    
    # 添加预取以提高性能
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # 计算每个epoch的步数
    total_injured = sum(len(status_dict['injured']) for organ, status_dict in organ_patients.items())
    steps_per_epoch = max(1, total_injured // batch_size)
    
    return dataset, steps_per_epoch




def create_tf_dataset(patient_ids, cache_dir, batch_size=4, shuffle=True):
    """创建tf.data.Dataset数据集，替代原有的Sequence生成器"""
    print("创建tf.data数据集...")
    
    # 过滤有效患者ID
    valid_patient_ids = []
    for pid in tqdm(patient_ids, desc="过滤有效患者ID"):
        image_file = os.path.join(cache_dir, f"{pid}.npz")
        label_file = os.path.join(cache_dir, f"{pid}_labels.npz")
        if os.path.exists(image_file) and os.path.exists(label_file):
            valid_patient_ids.append(pid)
    
    print(f"找到 {len(valid_patient_ids)} 个有效的缓存患者数据")
    
    # 创建一个tf.data.Dataset，包含所有有效患者ID
    patient_ds = tf.data.Dataset.from_tensor_slices(valid_patient_ids)
    
    # 如果需要洗牌
    if shuffle:
        patient_ds = patient_ds.shuffle(buffer_size=min(len(valid_patient_ids), 100), 
                                    reshuffle_each_iteration=True)
    
    # 加载患者数据
    def load_patient(patient_id):
        # 将字符串张量转换为Python字符串
        pid = patient_id.numpy().decode('utf-8')
        images, labels = load_patient_data_for_tf(pid, cache_dir)
        return images, labels
    
    # 使用tf.py_function包装Python函数，使其可以在tf.data管道中使用
    def load_patient_wrapper(patient_id):
        images, labels = tf.py_function(
            func=load_patient,
            inp=[patient_id],
            Tout=[tf.float32, tf.float32]
        )
        # 设置形状信息，因为py_function不会保留
        images.set_shape((32, 224, 224, 1))
        labels.set_shape((5, 4))
        return images, labels

    # 关键：先repeat再map，确保数据永远不会耗尽
    patient_ds = patient_ds.repeat()
    
    # 映射到加载函数
    dataset = patient_ds.map(
        load_patient_wrapper,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # 过滤无效数据
    def is_valid(images, labels):
        return tf.math.reduce_all(tf.math.is_finite(images))
    
    dataset = dataset.filter(is_valid)
    
    # 添加repeat()方法，使数据集可以在多个epoch中重复使用
    dataset = dataset.repeat()  # 无限重复
    
    # 批处理
    dataset = dataset.batch(batch_size)
    
    # 预取下一批数据
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # 启用性能优化
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    
    # 应用选项
    dataset = dataset.with_options(options)
    
    return dataset, len(valid_patient_ids)



# 可视化类别分布
def visualize_class_distribution(dataset, steps, title="Class Distribution"):
    """可视化数据集的类别分布"""
    # 收集数据
    organ_counts = {
        'bowel': {'healthy': 0, 'injury': 0},
        'extravasation': {'healthy': 0, 'injury': 0},
        'kidney': {'healthy': 0, 'injury': 0},
        'liver': {'healthy': 0, 'injury': 0},
        'spleen': {'healthy': 0, 'injury': 0}
    }
    
    total_samples = 0
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
    for x_batch, y_batch in tqdm(dataset.take(steps), total=steps, desc="Collecting data"):
        batch_size = x_batch.shape[0]
        total_samples += batch_size
        
        y_batch_np = y_batch.numpy()
        
        # 统计每个器官的健康和损伤样本
        for i, organ in enumerate(organs):
            healthy_count = np.sum(y_batch_np[:, i, 0])
            injury_count = np.sum(y_batch_np[:, i, 1])
            
            organ_counts[organ]['healthy'] += healthy_count
            organ_counts[organ]['injury'] += injury_count
    
    # 可视化
    plt.figure(figsize=(14, 8))
    
    # 绘制条形图
    x = np.arange(len(organs))
    width = 0.35
    
    healthy_percentages = [organ_counts[organ]['healthy'] / total_samples * 100 for organ in organs]
    injury_percentages = [organ_counts[organ]['injury'] / total_samples * 100 for organ in organs]
    
    plt.bar(x - width/2, healthy_percentages, width, label='Healthy')
    plt.bar(x + width/2, injury_percentages, width, label='Injured')
    
    plt.xlabel('Organ')
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.xticks(x, organs)
    plt.legend()
    
    # 添加数值标签
    for i, v in enumerate(healthy_percentages):
        plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(injury_percentages):
        plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{title.replace(" ", "_")}.png'), dpi=300)
    plt.show()





# 可视化训练历史记录的函数
def plot_training_history(history):
    """
    可视化模型训练历史
    
    Args:
        history: 模型训练历史对象
    """
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=300)
    plt.show()



def visualize_segmentation_results(unet_model, patient_id, meta_df, dicom_tags_df, segmentation_map, num_samples=5):
    """
    可视化U-Net分割模型的结果
    Args:
        unet_model: 训练好的U-Net模型
        patient_id: 患者ID
        meta_df: 元数据DataFrame
        dicom_tags_df: DICOM标签DataFrame
        segmentation_map: 分割映射
        num_samples: 要显示的样本数量 (增加到5)
    """
    try:
        # 获取患者的系列ID
        subset = meta_df[meta_df['patient_id'] == patient_id]
        if subset.empty:
            print(f"未找到患者 {patient_id} 的元数据")
            return

        series_id = str(subset['series_id'].iloc[0])

        # 获取图像ID
        image_ids = dicom_tags_df[
            (dicom_tags_df['PatientID'] == patient_id) &
            (dicom_tags_df['series_id'] == series_id)
        ]['InstanceNumber'].values

        if len(image_ids) == 0:
            print(f"未找到患者 {patient_id} 的图像ID")
            return

        image_ids = sorted(image_ids)

        # 获取分割数据
        segmentation_file = segmentation_map.get(patient_id)
        segmentation_data = None
        if segmentation_file and os.path.exists(segmentation_file):
            try:
                segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
            except Exception as e:
                print(f"加载分割数据时出错: {str(e)}")

        # 随机选择num_samples个切片
        if len(image_ids) <= num_samples:
            selected_ids = image_ids
        else:
            selected_ids = random.sample(list(image_ids), num_samples)

        plt.figure(figsize=(15, 5 * num_samples)) # 调整图形大小

        for i, image_id in enumerate(selected_ids):
            image_file = os.path.join(TRAIN_IMAGES, str(patient_id), str(series_id), f'{image_id}.dcm')

            # 加载图像
            dicom = pydicom.dcmread(image_file)
            image = dicom.pixel_array.astype(np.float32)
            image = cv2.resize(image, IMG_SIZE)
            max_val = np.max(image)
            image = image / max_val if max_val != 0 else image

            # 加载真实分割掩码
            ground_truth = np.zeros(IMG_SIZE, dtype=np.float32)
            if segmentation_data is not None and image_id <= segmentation_data.shape[2]:
                mask_slice = segmentation_data[:, :, image_id-1]
                ground_truth = cv2.resize(mask_slice, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            # 使用U-Net模型进行分割
            if len(image.shape) == 2:
                image_for_seg = np.stack([image, image, image], axis=-1)
            else:
                image_for_seg = image

            predicted_mask = segment_slice(image_for_seg, unet_model, IMG_SIZE)

            # 可视化结果
            plt.subplot(num_samples, 3, i * 3 + 1)  #  3 列
            plt.imshow(image, cmap='gray')
            plt.title(translate(f'原始图像 (ID: {image_id})'))
            plt.axis('off')

            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(ground_truth, cmap='jet', alpha=0.7)
            plt.title(translate('真实分割掩码'))
            plt.axis('off')

            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(image, cmap='gray')
            plt.imshow(predicted_mask, cmap='jet', alpha=0.7)
            plt.title(translate('预测分割掩码'))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'segmentation_results_{patient_id}.png'), dpi=300)
        plt.show()

    except Exception as e:
        print(f"可视化分割结果时出错: {str(e)}")




def visualize_classification_results(classifier_model, unet_model, patient_id, meta_df, dicom_tags_df, 
                                    segmentation_map, organ_status_map):
    """
    可视化分类模型的结果
    """
    try:
        # 获取患者的系列ID
        subset = meta_df[meta_df['patient_id'] == patient_id]
        if subset.empty:
            print(f"未找到患者 {patient_id} 的元数据")
            return
        
        series_id = str(subset['series_id'].iloc[0])
        
        # 获取图像ID
        image_ids = dicom_tags_df[
            (dicom_tags_df['PatientID'] == patient_id) &
            (dicom_tags_df['series_id'] == series_id)
        ]['InstanceNumber'].values
        
        if len(image_ids) == 0:
            print(f"未找到患者 {patient_id} 的图像ID")
            return
            
        image_ids = sorted(image_ids)
        
        # 获取分割数据
        segmentation_file = segmentation_map.get(patient_id)
        segmentation_data = None
        if segmentation_file and os.path.exists(segmentation_file):
            try:
                segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
            except Exception as e:
                print(f"加载分割数据时出错: {str(e)}")
        
        # 加载所有切片
        all_slices = []
        for image_id in image_ids:
            image_file = os.path.join(TRAIN_IMAGES, str(patient_id), str(series_id), f'{image_id}.dcm')
            try:
                dicom = pydicom.dcmread(image_file)
                image = dicom.pixel_array.astype(np.float32)
                image = cv2.resize(image, IMG_SIZE)
                max_val = np.max(image)
                image = image / max_val if max_val != 0 else image
                
                mask = np.zeros(IMG_SIZE, dtype=np.float32)
                if segmentation_data is not None and image_id <= segmentation_data.shape[2]:
                    mask_slice = segmentation_data[:, :, image_id-1]
                    mask = cv2.resize(mask_slice, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
                
                all_slices.append({
                    'image': image,
                    'mask': mask,
                    'instance_number': int(image_id)
                })
            except Exception as e:
                print(f"加载切片 {image_id} 时出错: {str(e)}")
                continue
        
        # 如果切片数量不足32，则退出
        if len(all_slices) < 32:
            print(f"患者 {patient_id} 的切片数量不足32，无法进行分类")
            return
        
        # 选择中间32张切片
        middle_index = len(all_slices) // 2
        start_index = max(0, middle_index - 16)
        sequence = all_slices[start_index:start_index + 32]
        
        # 处理切片
        processed_sequence = []
        for slice_data in sequence:
            image = slice_data['image']
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            
            segmented_slice = segment_slice(image, unet_model, IMG_SIZE)
            processed_sequence.append(np.expand_dims(segmented_slice, axis=-1))
        
        # 准备输入数据
        X = np.expand_dims(np.array(processed_sequence), axis=0)
        
        # 使用分类模型进行预测
        y_pred = classifier_model.predict(X)[0]
        
        # 获取真实标签
        organ_status = organ_status_map.get(patient_id, {})
        y_true = []
        organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        for organ in organs:
            status = organ_status.get(organ, {})
            organ_label = [
                status.get('healthy', 0),
                status.get('injury', 0),
                status.get('low', 0),
                status.get('high', 0)
            ]
            y_true.append(organ_label)
        
        y_true = np.array(y_true)
        
        # 可视化结果 - 修改部分
        plt.figure(figsize=(15, 10))
        
        # 显示中间的一张切片
        middle_slice = processed_sequence[16][:, :, 0]
        plt.subplot(2, 3, 1)
        plt.imshow(middle_slice, cmap='gray')
        plt.title('代表性切片')
        plt.axis('off')
        
        # 显示每个器官的预测结果
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        plt.subplot(2, 3, 2)
        for i, organ in enumerate(organs):
            plt.bar(i, y_pred[i, 1], color=colors[i])
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
        plt.xticks(range(len(organs)), organs, rotation=45)
        plt.ylim(0, 1)
        plt.title('损伤预测概率')
        
        # 显示真实标签和预测标签的比较
        plt.subplot(2, 3, 3)
        for i, organ in enumerate(organs):
            true_label = np.argmax(y_true[i])
            pred_label = np.argmax(y_pred[i])
            
            plt.scatter(i-0.1, true_label, color='blue', label='真实' if i == 0 else '')
            plt.scatter(i+0.1, pred_label, color='red', label='预测' if i == 0 else '')
        
        plt.xticks(range(len(organs)), organs, rotation=45)
        plt.yticks(range(4), ['健康', '损伤', '低度', '高度'])
        plt.title('真实标签 vs 预测标签')
        plt.legend()
        
        # 显示详细的器官预测结果 - 只显示前3个器官
        for i in range(min(3, len(organs))):
            plt.subplot(2, 3, 4 + i)  # 最大索引为6
            
            bar_width = 0.35
            index = np.arange(4)
            
            plt.bar(index, y_true[i], bar_width, label='真实', color='blue', alpha=0.7)
            plt.bar(index + bar_width, y_pred[i], bar_width, label='预测', color='red', alpha=0.7)
            
            plt.xlabel(organs[i].capitalize())
            plt.xticks(index + bar_width/2, ['健康', '损伤', '低度', '高度'])
            plt.ylim(0, 1)
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'classification_results_{patient_id}.png'), dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"可视化分类结果时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())



# 绘制ROC曲线
def plot_roc_curves(evaluation_results):
    """绘制ROC曲线"""
    plt.figure(figsize=(15, 10))
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
    for i, result in enumerate([r for r in evaluation_results if r['organ'] in organs]):
        plt.subplot(2, 3, i+1)
        organ = result['organ']
        
        # 获取ROC曲线数据
        fpr = result['fpr']
        tpr = result['tpr']
        auc_value = result['auc']
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {auc_value:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{organ.capitalize()} ROC Curve')
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300)
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrices(evaluation_results):
    """绘制混淆矩阵热图"""
    plt.figure(figsize=(15, 10))
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
    for i, result in enumerate([r for r in evaluation_results if r['organ'] in organs]):
        plt.subplot(2, 3, i+1)
        organ = result['organ']
        cm = result['confusion_matrix']
        
        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Healthy', 'Injured'], 
                   yticklabels=['Healthy', 'Injured'])
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'{organ.capitalize()} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=300)
    plt.show()



import time
import sys

class ProgressLogger:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        
    def log(self, message):
        current_time = time.time()
        elapsed = current_time - self.last_time
        total_elapsed = current_time - self.start_time
        
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(total_elapsed))
        
        print(f"[{time_str}] [{elapsed_str}] (+{elapsed:.2f}s) {message}")
        sys.stdout.flush()  # 确保立即显示
        
        self.last_time = current_time

# 创建全局日志记录器
logger = ProgressLogger()



class DetailedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir="logs", **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.start_time = None
        self.epoch_start_time = None
        self.batch_times = []
        
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.start_time = time.time()
        print("训练开始，时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        
    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.epoch_start_time = time.time()
        self.batch_times = []
        print(f"\n===== 开始 Epoch {epoch+1} =====")
        
    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        if batch % 10 == 0:  # 每10个批次打印一次
            current_time = time.time()
            
            # 获取steps_per_epoch，如果为None则显示为'?'
            steps_per_epoch = self.params.get('steps')
            steps_display = steps_per_epoch if steps_per_epoch is not None else '?'
            
            # 添加当前批次的时间
            if self.batch_times:
                last_time = self.batch_times[-1]
                batch_time = current_time - last_time
                self.batch_times.append(current_time)
            else:
                batch_time = current_time - self.epoch_start_time
                self.batch_times.append(current_time)
            
            # 计算平均批次时间和ETA
            if len(self.batch_times) > 1:
                # 计算相邻时间点的差值来获取批次时间
                recent_times = [self.batch_times[i] - self.batch_times[i-1] for i in range(1, min(11, len(self.batch_times)))]
                avg_time = sum(recent_times) / len(recent_times)
                
                # 安全检查：确保steps_per_epoch不是None，才进行ETA计算
                if steps_per_epoch is not None:
                    eta = avg_time * (steps_per_epoch - batch)
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
                    print(f"Batch {batch}/{steps_display}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.4f}, ETA: {eta_str}")
                else:
                    # 如果steps_per_epoch是None，只显示当前进度，不显示ETA
                    print(f"Batch {batch}/{steps_display}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.4f}")
            else:
                print(f"Batch {batch}/{steps_display}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.4f}")
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n===== Epoch {epoch+1} 完成 =====")
        print(f"训练损失: {logs.get('loss', 0):.4f}, 训练准确率: {logs.get('accuracy', 0):.4f}")
        print(f"验证损失: {logs.get('val_loss', 0):.4f}, 验证准确率: {logs.get('val_accuracy', 0):.4f}")
        print(f"Epoch 耗时: {epoch_time:.2f}秒")
        print(f"已训练时间: {(time.time() - self.start_time):.2f}秒")
        super().on_epoch_end(epoch, logs)
        
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        print("\n===== 训练完成 =====")
        print(f"总训练时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        super().on_train_end(logs)

class ClassBalanceMonitor(tf.keras.callbacks.Callback):
    def __init__(self, dataset, steps):
        super().__init__()
        self.dataset = dataset
        self.steps = steps
        self.organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        
    def on_epoch_begin(self, epoch, logs=None):
        # 检查训练批次中的类别分布
        print("\n检查训练数据类别分布:")
        
        # 初始化计数器
        healthy_counts = np.zeros(5)
        injury_counts = np.zeros(5)
        low_counts = np.zeros(5)
        high_counts = np.zeros(5)
        total_samples = 0
        
        # 只分析前5个批次，以免花费太多时间
        for x_batch, y_batch in self.dataset.take(5):  
            batch_size = x_batch.shape[0]
            total_samples += batch_size
            y_batch_np = y_batch.numpy()
            
            # 统计每个器官的各种损伤样本
            for i, organ in enumerate(self.organs):
                healthy_counts[i] += np.sum(y_batch_np[:, i, 0])
                injury_counts[i] += np.sum(y_batch_np[:, i, 1])
                low_counts[i] += np.sum(y_batch_np[:, i, 2])
                high_counts[i] += np.sum(y_batch_np[:, i, 3])
        
        # 打印详细统计信息
        print(f"分析了 {total_samples} 个样本:")
        for i, organ in enumerate(self.organs):
            print(f"  {organ}:")
            if total_samples > 0:
                healthy_pct = (healthy_counts[i] / total_samples) * 100
                injury_pct = (injury_counts[i] / total_samples) * 100
                low_pct = (low_counts[i] / total_samples) * 100
                high_pct = (high_counts[i] / total_samples) * 100
                
                print(f"    健康: {healthy_counts[i]:.0f} ({healthy_pct:.2f}%)")
                print(f"    损伤: {injury_counts[i]:.0f} ({injury_pct:.2f}%)")
                print(f"    低度: {low_counts[i]:.0f} ({low_pct:.2f}%)")
                print(f"    高度: {high_counts[i]:.0f} ({high_pct:.2f}%)")


def weighted_cross_entropy(organ_weights=None):
    """
    返回一个加权交叉熵损失函数，为不同器官设置不同权重
    """
    # 如果没有提供权重，使用基于类别不平衡的默认权重
    if organ_weights is None:
        # 基于论文中的数据分布计算权重
        # 肠道(Bowel)：损伤比例2.3%，权重约43
        # 渗出(Extravasation)：6.8%，权重约15
        # 肾脏(Kidney)：6.9%，权重约14.5
        # 肝脏(Liver)：10.8%，权重约9.3
        # 脾脏(Spleen)：11.8%，权重约8.5
        organ_weights = [43.0, 15.0, 14.5, 9.3, 8.5]
    
    def loss(y_true, y_pred):
        """计算加权交叉熵损失"""
        organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
        total_loss = 0.0
        
        for i, (organ, weight) in enumerate(zip(organs, organ_weights)):
            # 提取每个器官的真实标签和预测值
            y_true_organ = y_true[:, i, :]
            y_pred_organ = y_pred[:, i, :]
            
            # 计算交叉熵损失
            epsilon = 1e-7  # 防止log(0)
            
            # 单独处理每个类别
            healthy_true = y_true_organ[:, 0]
            injury_true = y_true_organ[:, 1]
            
            healthy_pred = y_pred_organ[:, 0]
            injury_pred = y_pred_organ[:, 1]
            
            # 基础交叉熵
            healthy_loss = -healthy_true * tf.math.log(healthy_pred + epsilon)
            injury_loss = -injury_true * tf.math.log(injury_pred + epsilon)
            
            # 应用权重 - 损伤类别权重更高
            weighted_loss = healthy_loss + weight * injury_loss
            
            # 添加到总损失
            total_loss += tf.reduce_mean(weighted_loss)
        
        return total_loss / len(organs)
    
    return loss



# 评估指标计算
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """计算与论文一致的评估指标"""
    # 将预测概率转换为二进制标签
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # 处理可能的除零错误
    if np.sum(y_true) > 0:  # 如果有正样本
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
    else:
        precision = 0
        recall = 0
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 从混淆矩阵中计算TP, TN, FP, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # 计算特异性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # 计算PPV和NPV
        ppv = precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    else:
        # 处理只有一个类别的情况
        if np.all(y_true == 0):  # 全是负样本
            tn = np.sum(y_pred_binary == 0)
            fp = np.sum(y_pred_binary == 1)
            fn = 0
            tp = 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 1
            ppv = 0
            npv = tn / tn if tn > 0 else 1
        else:  # 全是正样本
            tn = 0
            fp = 0
            fn = np.sum(y_pred_binary == 0)
            tp = np.sum(y_pred_binary == 1)
            specificity = 0
            ppv = tp / tp if tp > 0 else 1
            npv = 0
    
    return accuracy, precision, recall, specificity, ppv, npv, cm

# 按器官评估模型
def evaluate_model_by_organ(model, val_dataset, validation_steps):
    """按器官单独评估模型性能"""
    organs = ['bowel', 'extravasation', 'kidney', 'liver', 'spleen']
    
    # 收集验证集预测
    all_true = []
    all_pred = []
    
    for x_batch, y_batch in tqdm(val_dataset.take(validation_steps), total=validation_steps, desc="Evaluating model"):
        y_pred = model.predict(x_batch, verbose=0)
        all_true.append(y_batch.numpy())
        all_pred.append(y_pred)
    
    # 合并批次结果
    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)
    
    # 按器官评估
    results = []
    for i, organ in enumerate(organs):
        # 提取当前器官的标签和预测
        organ_true = y_true[:, i, 1]  # 只考虑injury标签
        organ_pred = y_pred[:, i, 1]
        
        # 计算评估指标
        accuracy, precision, recall, specificity, ppv, npv, cm = calculate_metrics(organ_true, organ_pred)
        
        # 计算ROC曲线和AUC
        try:
            fpr, tpr, _ = roc_curve(organ_true, organ_pred)
            auc_value = auc(fpr, tpr)
        except:
            fpr, tpr = [0, 1], [0, 1]
            auc_value = 0.5
        
        # 记录结果
        results.append({
            'organ': organ,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'auc': auc_value,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        })
        
        # 打印结果
        print(f"\nOrgan: {organ}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  PPV: {ppv:.4f}")
        print(f"  NPV: {npv:.4f}")
        print(f"  AUC: {auc_value:.4f}")
        print(f"  Confusion Matrix: \n{cm}")
    
    return results





def get_lr_schedule():
    """创建余弦退火学习率调度器"""
    initial_learning_rate = 3e-4
    
    def cosine_decay_with_warmup(epoch):
        # 前2个epoch为预热期，逐渐增加学习率
        if epoch < 2:
            return initial_learning_rate * ((epoch + 1) / 2)
        
        # 之后使用余弦衰减
        decay_epochs = 15 - 2  # 总epochs减去预热epochs
        epoch_in_decay_range = epoch - 2  # 调整epoch计数
        
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch_in_decay_range / decay_epochs))
        return initial_learning_rate * cosine_decay
    
    return tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup)



# 5k-fold交叉验证实现
def train_with_kfold_cv(train_df, organ_status_map, segmentation_map, unet_model, k=5):
    """
    使用k-fold交叉验证训练和评估模型
    
    Args:
        train_df: 训练数据DataFrame
        organ_status_map: 器官状态映射
        segmentation_map: 分割映射
        unet_model: U-Net模型
        k: 交叉验证折数
    
    Returns:
        训练好的模型列表和评估结果
    """
    logger.log(f"Starting {k}-fold cross-validation...")
    
    # 获取所有患者ID
    patient_ids = np.array(train_df['patient_id'].unique())
    
    # 创建k-fold拆分器
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # 加载元数据
    meta_df = pd.read_csv(TRAIN_META)
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    
    # 加载DICOM标签
    logger.log("Loading DICOM tags...")
    dicom_tags_df = pd.read_parquet(TRAIN_DICOM_TAGS)
    temp = dicom_tags_df['SeriesInstanceUID'].str.split('.', expand=True)
    dicom_tags_df['series_id'] = temp[8]
    dicom_tags_df['series_id'] = dicom_tags_df['series_id'].astype(str)
    dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
    gc.collect()
    
    # 存储每个fold的模型和结果
    fold_models = []
    fold_results = []
    
    # 定义批量大小
    batch_size = 4
    
    # 遍历每个fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(patient_ids)):
        logger.log(f"\n{'='*50}\nFold {fold+1}/{k}\n{'='*50}")
        
        # 获取当前fold的训练和验证患者ID
        train_patient_ids = patient_ids[train_idx]
        val_patient_ids = patient_ids[val_idx]
        
        logger.log(f"Train set: {len(train_patient_ids)} patients, Validation set: {len(val_patient_ids)} patients")
        
        # 为当前fold创建缓存目录
        fold_cache_dir = os.path.join(CACHE_DIR, f"fold_{fold+1}")
        os.makedirs(fold_cache_dir, exist_ok=True)
        
        # 预处理患者数据
        logger.log(f"Preprocessing patients for fold {fold+1}...")
        preprocess_patients_parallel(
            train_patient_ids, meta_df, dicom_tags_df, segmentation_map, 
            organ_status_map, unet_model, fold_cache_dir, 
            img_size=IMG_SIZE, max_slices=32, batch_size=10
        )
        
        preprocess_patients_parallel(
            val_patient_ids, meta_df, dicom_tags_df, segmentation_map, 
            organ_status_map, unet_model, fold_cache_dir, 
            img_size=IMG_SIZE, max_slices=32, batch_size=10
        )
        
        # 创建数据集
        logger.log(f"Creating datasets for fold {fold+1}...")
        train_dataset, train_steps = create_organ_balanced_dataset(
            train_patient_ids, fold_cache_dir, batch_size=batch_size, shuffle=True
        )
        
        val_dataset, num_val_samples = create_tf_dataset(
            val_patient_ids, fold_cache_dir, batch_size=batch_size, shuffle=False
        )
        
        val_steps = num_val_samples // batch_size
        
        # 构建分类器模型
        logger.log(f"Building classifier model for fold {fold+1}...")
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)  # mask为1通道
        loss_fn = weighted_cross_entropy()  # 使用加权交叉熵损失函数
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        
        classifier_model = build_25d_classifier(input_shape=input_shape, num_organs=NUM_ORGANS)
        classifier_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        # 设置回调函数
        logger.log(f"Setting up callbacks for fold {fold+1}...")
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, f'classifier_fold_{fold+1}_best.keras'),
            monitor='val_loss', save_best_only=True
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6
        )
        
        # 使用我们定义的学习率调度器
        lr_scheduler = get_lr_schedule()
        
        tensorboard_callback = DetailedTensorBoard(
            log_dir=f"./logs/classifier_fold_{fold+1}", histogram_freq=1
        )
        
        callbacks = [model_checkpoint, early_stopping, reduce_lr, lr_scheduler, tensorboard_callback]
        
        # 训练模型
        logger.log(f"Training classifier for fold {fold+1}...")
        history = classifier_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=15,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存模型
        classifier_model.save(os.path.join(OUTPUT_DIR, f'classifier_fold_{fold+1}.keras'))
        
        # 评估模型
        logger.log(f"Evaluating classifier for fold {fold+1}...")
        evaluation_results = evaluate_model_by_organ(classifier_model, val_dataset, val_steps)
        
        # 保存评估结果
        results_df = pd.DataFrame([
            {
                'fold': fold + 1,
                'organ': result['organ'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'specificity': result['specificity'],
                'ppv': result['ppv'],
                'npv': result['npv'],
                'auc': result['auc']
            }
            for result in evaluation_results
        ])
        
        results_df.to_csv(os.path.join(OUTPUT_DIR, f'evaluation_results_fold_{fold+1}.csv'), index=False)
        
        # 绘制ROC曲线
        plot_roc_curves(evaluation_results)
        plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curves_fold_{fold+1}.png'), dpi=300)
        
        # 绘制混淆矩阵
        plot_confusion_matrices(evaluation_results)
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrices_fold_{fold+1}.png'), dpi=300)
        
        # 存储模型和结果
        fold_models.append(classifier_model)
        fold_results.append(evaluation_results)
        
        # 清理内存
        gc.collect()
    
    # 计算所有fold的平均性能
    logger.log("Calculating average performance across all folds...")
    
    # 提取所有fold的指标
    all_metrics = []
    for fold, results in enumerate(fold_results):
        for result in results:
            all_metrics.append({
                'fold': fold + 1,
                'organ': result['organ'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'specificity': result['specificity'],
                'ppv': result['ppv'],
                'npv': result['npv'],
                'auc': result['auc']
            })
    
    # 创建DataFrame并计算平均值
    all_metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = all_metrics_df.groupby('organ').mean().reset_index()
    avg_metrics['fold'] = 'Average'
    
    # 保存平均指标
    avg_metrics.to_csv(os.path.join(OUTPUT_DIR, 'average_metrics.csv'), index=False)
    
    # 打印平均性能
    print("\nAverage Performance Across All Folds:")
    for _, row in avg_metrics.iterrows():
        organ = row['organ']
        print(f"\nOrgan: {organ}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  AUC: {row['auc']:.4f}")
        print(f"  Precision: {row['precision']:.4f}")
        print(f"  Recall (Sensitivity): {row['recall']:.4f}")
        print(f"  Specificity: {row['specificity']:.4f}")
        print(f"  PPV: {row['ppv']:.4f}")
        print(f"  NPV: {row['npv']:.4f}")
    
    return fold_models, fold_results



# 主函数 - 修改为使用外部缓存和预训练模型
def main():
    logger = ProgressLogger()
    logger.log("Starting main function")
    
    # 重新定义缓存路径，使用已有的缓存
    ORIGINAL_CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')  # 原始定义
    EXTERNAL_CACHE_DIR = '/kaggle/input/rsna-cache/cache'  # 外部已有缓存
    
    # 检查外部缓存是否存在
    if os.path.exists(EXTERNAL_CACHE_DIR):
        logger.log(f"Found external cache: {EXTERNAL_CACHE_DIR}")
        CACHE_DIR = EXTERNAL_CACHE_DIR  # 使用外部缓存
    else:
        logger.log(f"External cache not found, will use original cache path: {ORIGINAL_CACHE_DIR}")
        CACHE_DIR = ORIGINAL_CACHE_DIR
    
    # 1. 数据加载与预处理
    logger.log("Loading training data...")
    train_df, organ_status_map, segmentation_map = load_and_process_data()
    
    # 设置5k-fold交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    patient_ids = train_df['patient_id'].unique()
    
    meta_df = pd.read_csv(TRAIN_META)
    meta_df['patient_id'] = meta_df['patient_id'].astype(str)
    
    logger.log("Loading DICOM tags...")
    dicom_tags_df = pd.read_parquet(TRAIN_DICOM_TAGS)
    temp = dicom_tags_df['SeriesInstanceUID'].str.split('.', expand=True)
    dicom_tags_df['series_id'] = temp[8]
    dicom_tags_df['series_id'] = dicom_tags_df['series_id'].astype(str)
    dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
    gc.collect()
    logger.log("Data loading complete")

    # 定义全局批量大小
    batch_size = 4

    # 2. U-Net模型训练或加载
    unet_input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    
    # 检查是否有预训练U-Net模型 - 按优先级顺序尝试不同路径
    unet_model_paths = [
        '/kaggle/input/unet-model/unet_model.keras',   # 外部上传模型
        '/kaggle/input/unet_model/pytorch/default/1/unet_model.keras',  # 另一个可能的路径
        'unet_best.keras',  # 最佳模型
        'unet_model.keras'  # 标准模型
    ]
    
    unet_model = None
    for model_path in unet_model_paths:
        if os.path.exists(model_path):
            logger.log(f"Loading pre-trained U-Net model from: {model_path}")
            try:
                unet_model = tf.keras.models.load_model(model_path)
                logger.log("U-Net model loaded successfully!")

                # 在这里调用可视化函数
                sample_patient_id = patient_ids[0]  # 使用第一个训练集患者
                visualize_segmentation_results(unet_model, sample_patient_id, meta_df, dicom_tags_df, segmentation_map, num_samples=3)
                
                break
            except Exception as e:
                logger.log(f"Failed to load model {model_path}: {str(e)}")
                continue
    
    if unet_model is None:
        logger.log("No pre-trained model found, will train a new one...")
        unet_model = build_unet(unet_input_shape)
        unet_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 为U-Net训练创建tf.data.Dataset
        logger.log("Creating U-Net training dataset...")
        
        # 定义加载切片的函数
        def load_slice_for_unet(patient_id, meta_df, dicom_tags_df, segmentation_map):
            try:
                # 将字符串张量转换为Python字符串
                pid = patient_id.numpy().decode('utf-8')
                
                # 获取患者元数据
                subset = meta_df[meta_df['patient_id'] == pid]
                if subset.empty:
                    return np.zeros((224, 224, 3), dtype=np.float32), np.zeros((224, 224, 1), dtype=np.float32)
                
                series_id = str(subset['series_id'].iloc[0])
                
                # 获取图像ID列表
                image_ids = dicom_tags_df[
                    (dicom_tags_df['PatientID'] == pid) &
                    (dicom_tags_df['series_id'] == series_id)
                ]['InstanceNumber'].values
                
                if len(image_ids) == 0:
                    return np.zeros((224, 224, 3), dtype=np.float32), np.zeros((224, 224, 1), dtype=np.float32)
                
                # 随机选择一个切片
                image_id = np.random.choice(image_ids)
                image_file = os.path.join(TRAIN_IMAGES, str(pid), str(series_id), f'{image_id}.dcm')
                
                # 加载图像
                dicom = pydicom.dcmread(image_file)
                image = dicom.pixel_array.astype(np.float32)
                image = cv2.resize(image, IMG_SIZE)
                max_val = np.max(image)
                image = image / max_val if max_val != 0 else image
                
                # 确保图像是3通道的
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)
                
                # 加载分割掩码
                mask = np.zeros(IMG_SIZE, dtype=np.float32)
                segmentation_file = segmentation_map.get(pid)
                if segmentation_file and os.path.exists(segmentation_file):
                    try:
                        segmentation_data = nib.load(segmentation_file).get_fdata().astype(np.float32)
                        if image_id <= segmentation_data.shape[2]:
                            mask_slice = segmentation_data[:, :, image_id-1]
                            mask = cv2.resize(mask_slice, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
                    except:
                        pass
                
                # 数据增强
                image, mask = augment_data(image, mask)
                
                # 确保掩码是单通道的
                if len(mask.shape) == 2:
                    mask = np.expand_dims(mask, axis=-1)
                
                return image, mask
            except:
                return np.zeros((224, 224, 3), dtype=np.float32), np.zeros((224, 224, 1), dtype=np.float32)
        
        # 创建tf.data.Dataset
        def create_unet_dataset(patient_ids, batch_size=4, shuffle=True):
            # 创建一个包含患者ID的数据集
            patient_ds = tf.data.Dataset.from_tensor_slices(patient_ids)
            
            # 如果需要洗牌
            if shuffle:
                patient_ds = patient_ds.shuffle(buffer_size=len(patient_ids), reshuffle_each_iteration=True)
            
            # 加载切片
            def load_slice_wrapper(patient_id):
                image, mask = tf.py_function(
                    func=lambda x: load_slice_for_unet(x, meta_df, dicom_tags_df, segmentation_map),
                    inp=[patient_id],
                    Tout=[tf.float32, tf.float32]
                )
                # 设置形状信息
                image.set_shape((224, 224, 3))
                mask.set_shape((224, 224, 1))
                return image, mask
            
            # 映射到加载函数
            dataset = patient_ds.map(
                load_slice_wrapper,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            # 过滤无效数据
            def is_valid(image, mask):
                return tf.math.reduce_all(tf.math.is_finite(image))
            
            dataset = dataset.filter(is_valid)
            
            # 批处理
            dataset = dataset.batch(batch_size)
            
            # 预取下一批数据
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        # 从train_df中获取所有患者ID并划分训练和验证集
        from sklearn.model_selection import train_test_split
        train_patient_ids, val_patient_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
        logger.log(f"Dataset split complete: {len(train_patient_ids)} training samples, {len(val_patient_ids)} validation samples")
        
        # 创建训练和验证数据集
        logger.log("Creating U-Net training and validation datasets...")
        unet_train_dataset = create_unet_dataset(train_patient_ids, batch_size=batch_size, shuffle=True)
        unet_val_dataset = create_unet_dataset(val_patient_ids, batch_size=batch_size, shuffle=False)
        
        # 计算steps_per_epoch和validation_steps
        unet_steps_per_epoch = len(train_patient_ids) // batch_size
        unet_validation_steps = len(val_patient_ids) // batch_size
        
        gc.collect()
        
        # 定义回调
        logger.log("Setting up U-Net training callbacks...")
        unet_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'unet_best.keras', monitor='val_loss', save_best_only=True
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )
        
        tensorboard_callback = DetailedTensorBoard(
            log_dir="./logs/unet", histogram_freq=1
        )
        
        # 使用tf.data API进行训练
        logger.log("Starting U-Net model training...")
        history_unet = unet_model.fit(
            unet_train_dataset,
            validation_data=unet_val_dataset,
            epochs=5,
            steps_per_epoch=unet_steps_per_epoch,  # 明确指定每个epoch的步数
            validation_steps=unet_validation_steps,  # 明确指定验证步数
            callbacks=[unet_checkpoint, early_stopping, tensorboard_callback],
            verbose=1
        )
        
        logger.log("Saving U-Net model...")
        unet_model.save('unet_model.keras')
        
        # 可视化训练历史记录
        logger.log("Visualizing U-Net training history...")
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_unet.history['loss'], label='Training Loss')
        plt.plot(history_unet.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_unet.history['accuracy'], label='Training Accuracy')
        plt.plot(history_unet.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'unet_training_history.png'), dpi=300)
        plt.show()
        
        del unet_train_dataset, unet_val_dataset
        gc.collect()
    
    # 3. 检查缓存文件并决定是否需要预处理
    # 检查cache文件是否完整
    def check_cache_files(patient_ids, cache_dir):
        valid_cached_ids = []
        missing_ids = []
        for pid in tqdm(patient_ids, desc="Checking cache files"):
            image_file = os.path.join(cache_dir, f"{pid}.npz")
            label_file = os.path.join(cache_dir, f"{pid}_labels.npz")
            if os.path.exists(image_file) and os.path.exists(label_file):
                valid_cached_ids.append(pid)
            else:
                missing_ids.append(pid)
        return valid_cached_ids, missing_ids

    # 实现5k-fold交叉验证
    logger.log("Implementing 5-fold cross-validation...")
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(patient_ids)):
        logger.log(f"Starting fold {fold+1}/5")
        
        # 获取当前fold的训练和验证患者ID
        train_patient_ids = patient_ids[train_idx]
        val_patient_ids = patient_ids[val_idx]
        
        # 为当前fold创建缓存目录
        fold_cache_dir = os.path.join(CACHE_DIR, f"fold_{fold+1}")
        os.makedirs(fold_cache_dir, exist_ok=True)
        
        # 检查当前fold的cache文件
        logger.log(f"Checking cache files for fold {fold+1}...")
        cached_train_ids, missing_train_ids = check_cache_files(train_patient_ids, fold_cache_dir)
        cached_val_ids, missing_val_ids = check_cache_files(val_patient_ids, fold_cache_dir)

        logger.log(f"Found {len(cached_train_ids)}/{len(train_patient_ids)} cached training samples")
        logger.log(f"Found {len(cached_val_ids)}/{len(val_patient_ids)} cached validation samples")
        
        # 如果有缺失的缓存文件，并且我们可以写入缓存目录
        if (len(missing_train_ids) > 0 or len(missing_val_ids) > 0) and os.access(fold_cache_dir, os.W_OK):
            logger.log(f"Need to generate cache for {len(missing_train_ids) + len(missing_val_ids)} samples")
            
            # 确保缓存目录存在
            os.makedirs(fold_cache_dir, exist_ok=True)
            
            # 只为缺失的样本生成缓存
            if len(missing_train_ids) > 0:
                logger.log(f"Preprocessing {len(missing_train_ids)} training samples...")
                preprocess_patients_parallel(
                    missing_train_ids, meta_df, dicom_tags_df, segmentation_map, 
                    organ_status_map, unet_model, fold_cache_dir, 
                    img_size=IMG_SIZE, max_slices=32, batch_size=10
                )
            
            if len(missing_val_ids) > 0:
                logger.log(f"Preprocessing {len(missing_val_ids)} validation samples...")
                preprocess_patients_parallel(
                    missing_val_ids, meta_df, dicom_tags_df, segmentation_map, 
                    organ_status_map, unet_model, fold_cache_dir, 
                    img_size=IMG_SIZE, max_slices=32, batch_size=10
                )
        elif len(missing_train_ids) > 0 or len(missing_val_ids) > 0:
            logger.log(f"Warning: Cache incomplete but {fold_cache_dir} is not writable")
            logger.log("Will use available cache files, but model performance may be affected")
            
            # 更新训练和验证ID列表，只使用有缓存的ID
            train_patient_ids = cached_train_ids
            val_patient_ids = cached_val_ids
            
            if len(train_patient_ids) == 0 or len(val_patient_ids) == 0:
                logger.log(f"Skipping fold {fold+1} due to insufficient cached data")
                continue
        else:
            logger.log("All data already cached, no preprocessing needed")
        
        # 4. 创建数据集
        # 使用平衡采样创建训练数据集
        logger.log(f"Creating training and validation datasets for fold {fold+1}...")
        train_dataset, train_steps = create_organ_balanced_dataset(train_patient_ids, fold_cache_dir, batch_size=batch_size, shuffle=True)
        
        # 验证集使用标准数据集，保持原始分布
        val_dataset, num_val_samples = create_tf_dataset(val_patient_ids, fold_cache_dir, batch_size=batch_size, shuffle=False)
        
        # 计算分类器训练的steps_per_epoch和validation_steps
        classifier_steps_per_epoch = train_steps
        classifier_validation_steps = num_val_samples // batch_size
        
        logger.log(f"Classifier training steps: {classifier_steps_per_epoch} steps/epoch, validation steps: {classifier_validation_steps} steps/epoch")
        
        # 5. 构建和训练分类器模型
        logger.log(f"Building classifier model for fold {fold+1}...")
        NUM_ORGANS = 5
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 1)  # mask为1通道
        loss_fn = weighted_cross_entropy()  # 使用加权交叉熵损失函数
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)  # 使用适当的学习率

        # 创建新模型
        classifier_model = build_25d_classifier(input_shape=input_shape, num_organs=NUM_ORGANS)
        classifier_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
            
        classifier_model.summary()

        # 设置回调函数
        logger.log(f"Setting up classifier training callbacks for fold {fold+1}...")
        classifier_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'classifier_fold_{fold+1}_best.keras',
            monitor='val_loss', save_best_only=True
        )
        
        classifier_logger = tf.keras.callbacks.CSVLogger(f"classifier_fold_{fold+1}_train.log")
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6
        )
        
        # 使用我们定义的学习率调度器
        lr_scheduler = get_lr_schedule()
        
        # 创建TensorBoard回调
        tensorboard_callback = DetailedTensorBoard(
            log_dir=f"./logs/classifier_fold_{fold+1}", histogram_freq=1
        )
        
        # 创建类别平衡监控器
        balance_monitor = ClassBalanceMonitor(train_dataset, min(5, classifier_steps_per_epoch))
        
        # 组合所有回调
        callbacks = [
            classifier_checkpoint, 
            classifier_logger,
            early_stopping,
            reduce_lr,
            lr_scheduler,
            tensorboard_callback,
            balance_monitor
        ]
        
        # 可视化类别分布
        visualize_class_distribution(train_dataset, min(10, classifier_steps_per_epoch), f"Training Set Class Distribution (After Balancing) - Fold {fold+1}")
        visualize_class_distribution(val_dataset, min(10, classifier_validation_steps), f"Validation Set Class Distribution - Fold {fold+1}")
        
        # 训练分类器模型
        logger.log(f"Starting classifier training for fold {fold+1}...")
        history_classifier = classifier_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=15,  # 增加到15轮
            steps_per_epoch=classifier_steps_per_epoch,
            validation_steps=classifier_validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.log(f"Saving classifier model for fold {fold+1}...")
        classifier_model.save(f'classifier_fold_{fold+1}.keras')
        
        # 可视化训练历史记录
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history_classifier.history['loss'], label='Training Loss')
        plt.plot(history_classifier.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history_classifier.history['accuracy'], label='Training Accuracy')
        plt.plot(history_classifier.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'training_history_fold_{fold+1}.png'), dpi=300)
        plt.show()
        
        # 6. 分类模型评估
        logger.log(f"Evaluating classifier model performance for fold {fold+1}...")
        evaluation_results = evaluate_model_by_organ(classifier_model, val_dataset, classifier_validation_steps)
        
        # 绘制ROC曲线
        plot_roc_curves(evaluation_results)
        
        # 绘制混淆矩阵
        plot_confusion_matrices(evaluation_results)
        
        # 保存评估结果
        results_df = pd.DataFrame([
            {
                'fold': fold + 1,
                'organ': result['organ'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'specificity': result['specificity'],
                'ppv': result['ppv'],
                'npv': result['npv'],
                'auc': result['auc']
            }
            for result in evaluation_results
        ])
        results_df.to_csv(os.path.join(OUTPUT_DIR, f'evaluation_results_fold_{fold+1}.csv'), index=False)
        logger.log(f"Evaluation results saved to {os.path.join(OUTPUT_DIR, f'evaluation_results_fold_{fold+1}.csv')}")
        
        # 添加到总结果
        fold_results.append(evaluation_results)
        
        # 7. 可视化分类结果
        logger.log(f"Visualizing classification results for fold {fold+1}...")
        # 随机选择3个患者进行可视化
        if isinstance(val_patient_ids, np.ndarray):
            sample_patients = random.sample(val_patient_ids.tolist(), min(3, len(val_patient_ids)))
        else:
            sample_patients = random.sample(list(val_patient_ids), min(3, len(val_patient_ids)))
        
        for patient_id in sample_patients:
            try:
                visualize_classification_results(classifier_model, unet_model, patient_id, meta_df, 
                                            dicom_tags_df, segmentation_map, organ_status_map)
            except Exception as e:
                logger.log(f"Error visualizing results for patient {patient_id}: {str(e)}")
        
        # 清理内存
        gc.collect()
    
    # 计算所有fold的平均性能
    logger.log("Calculating average performance across all folds...")
    
    # 提取所有fold的指标
    all_metrics = []
    for fold, results in enumerate(fold_results):
        for result in results:
            all_metrics.append({
                'fold': fold + 1,
                'organ': result['organ'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'specificity': result['specificity'],
                'ppv': result['ppv'],
                'npv': result['npv'],
                'auc': result['auc']
            })
    
    # 创建DataFrame并计算平均值
    all_metrics_df = pd.DataFrame(all_metrics)
    avg_metrics = all_metrics_df.groupby('organ').mean().reset_index()
    avg_metrics['fold'] = 'Average'
    
    # 保存平均指标
    avg_metrics.to_csv(os.path.join(OUTPUT_DIR, 'average_metrics.csv'), index=False)
    
    # 打印平均性能
    print("\nAverage Performance Across All Folds:")
    print("=" * 60)
    for _, row in avg_metrics.iterrows():
        organ = row['organ']
        print(f"\nOrgan: {organ}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  AUC: {row['auc']:.4f}")
        print(f"  Precision: {row['precision']:.4f}")
        print(f"  Recall (Sensitivity): {row['recall']:.4f}")
        print(f"  Specificity: {row['specificity']:.4f}")
        print(f"  PPV: {row['ppv']:.4f}")
        print(f"  NPV: {row['npv']:.4f}")
    print("=" * 60)
    
    # 生成最终的结果表格
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 隐藏轴线
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格数据
    table_data = []
    for _, row in avg_metrics.iterrows():
        table_data.append([
            row['organ'].capitalize(),
            f"{row['auc']:.3f}",
            f"{row['accuracy']:.3f}",
            f"{row['ppv']:.3f}",
            f"{row['npv']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['specificity']:.3f}"
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['Organ', 'AUC', 'Accuracy', 'PPV', 'NPV', 'Sensitivity', 'Specificity'],
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * 7
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 设置标题
    plt.title('Average Performance Metrics Across All Folds', fontsize=14, pad=20)
    plt.tight_layout()
    
    # 保存表格
    plt.savefig(os.path.join(OUTPUT_DIR, 'performance_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.log("Training and evaluation complete!")
    gc.collect()
    
    return logger  # 返回日志记录器以便在主脚本中使用


if __name__ == "__main__":
    # 设置TensorFlow日志级别，只显示警告和错误
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=全部显示, 1=不显示INFO, 2=不显示INFO和WARNING
    
    # 清理内存
    gc.collect()
    
    # 创建日志记录器
    logger = ProgressLogger()
    
    # 记录开始时间
    start_time = time.time()
    logger.log("Program execution started")
    
    try:
        # 执行主函数，获取返回的logger
        main_logger = main()
        
        # 如果main函数返回了logger，使用它；否则使用当前logger
        active_logger = main_logger if main_logger else logger
    except Exception as e:
        logger.log(f"Error during execution: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise
    finally:
        # 记录结束时间
        end_time = time.time()
        elapsed = end_time - start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.log(f"Program execution complete, total time: {int(hours)} hours {int(minutes)} minutes {seconds:.2f} seconds")
        
        # 最终清理内存
        gc.collect()





