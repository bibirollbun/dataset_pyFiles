# Cell 0: 设置 Matplotlib 中文字体 (Kaggle 环境) 
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


# -*- coding: utf-8 -*-
# === 导入必要的库 ===
import concurrent.futures
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns # 根据需要取消注释
from tqdm.notebook import tqdm
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split # 如果需要K折交叉验证，可能需要 KFold 或 GroupKFold
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import random
import glob
import nibabel as nib
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# import matplotlib as mpl # 根据需要取消注释

# === 配置 ===
# --- 数据路径 ---
DATA_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
SEGMENTATION_DIR = os.path.join(DATA_DIR, 'segmentations') # 确认这个路径正确
PREPROCESSED_DIR = '/kaggle/input/rsna-uneted/preprocessed_data' # 新增：存储预处理数据的目录

# --- 输出路径 ---
OUTPUT_DIR = '/kaggle/working/'
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'unet_model_v2')
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'segmentation_predictions_multi_v2')

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)

# --- 图像和模型参数 ---
IMG_SIZE = (224, 224)
TARGET_SIZE = IMG_SIZE[0]
TARGET_SIZE_INT = IMG_SIZE[0]
N_INPUT_CHANNELS = 3
BEST_NIFTI_ORIENTATION_TRANSFORM = ['rot90']
USE_REVERSE_NIFTI_MAPPING = True # 设置为True来启用反向映射

# 定义器官映射 (修正后，合并左右肾)
# NII标签: 1:肝脏, 2:脾脏, 3:左肾, 4:右肾, 5:肠道
ORGAN_MAP_NII = {
    1: 'liver',
    2: 'spleen',
    3: 'kidney',  # 合并标签 3 和 4
    5: 'bowel'
}
# 输出通道映射 (模型输出顺序)
ORGAN_CHANNEL_MAP = {
    'liver': 0,
    'spleen': 1,
    'kidney': 2,
    'bowel': 3
}
NUM_ORGANS = len(ORGAN_CHANNEL_MAP) # 现在是 4

print(f"分割目标数量: {NUM_ORGANS}")
print(f"器官 -> NII值 映射 (处理方式): {ORGAN_MAP_NII}")
print(f"器官 -> 模型输出通道 映射: {ORGAN_CHANNEL_MAP}")

OUTPUT_MODEL_FILENAME = f'unet_effb0_multi_organ_{TARGET_SIZE}px_v2.keras'
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, OUTPUT_MODEL_FILENAME) 
print(f"最终最佳模型将保存到 (可写路径): {MODEL_SAVE_PATH}")

# --- 训练参数 ---
VALIDATION_SPLIT = 0.15 # 考虑使用 K-Fold 交叉验证以更好地复现论文
RANDOM_STATE = 42
BATCH_SIZE = 8  # 增大批量大小以提高训练速度
EPOCHS_STAGE1 = 20 # 可以根据需要调整各阶段Epochs
EPOCHS_STAGE2 = 15
EPOCHS_STAGE3 = 15
EPOCHS_STAGE4 = 20 # 最后阶段可以多训练一些
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-6

# --- 推理参数 ---
INFERENCE_BATCH_SIZE = 32  # 增大批量大小以提高推理速度
PREDICTION_THRESHOLD = 0.5

# --- 预处理参数 ---
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
PARALLEL_CALLS = tf.data.AUTOTUNE
CACHE_DATASET = False



# === 设置随机种子 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子设置为: {seed}")

set_seed(RANDOM_STATE)

# === 辅助函数 ===
def load_dicom_slice(path):
    """加载单个DICOM切片，应用VOI LUT，归一化，并获取InstanceNumber"""
    try:
        dicom_file = pydicom.dcmread(path)
        instance_number = int(dicom_file.InstanceNumber)
        image = apply_voi_lut(dicom_file.pixel_array, dicom_file)

        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image) # 处理全黑或全白图像

        image = image.astype(np.float32)

        # 处理 MONOCHROME1 (图像像素值需要反转)
        if 'PhotometricInterpretation' in dicom_file and dicom_file.PhotometricInterpretation == "MONOCHROME1":
             image = 1.0 - image

        return image, instance_number
    except Exception as e:
        # print(f"加载 DICOM 错误 {path}: {e}") # 可以取消注释以调试
        return None, None


    
# 在您的 Cell 3 (随机种子&辅助函数) 中修改或添加

def apply_orientation_transform(mask_slice, transform_ops=None):
    """
    应用方向变换到掩码切片。
    transform_ops: 一个包含操作字符串的列表，例如 ['rot90', 'fliplr']
                   可能的op: 'rot90', 'rot90_2', 'rot90_3', 'fliplr', 'flipud'
    """
    if transform_ops is None:
        return mask_slice

    transformed_mask = mask_slice.copy()
    for op in transform_ops:
        if op == 'rot90':
            transformed_mask = np.rot90(transformed_mask)
        elif op == 'rot90_2': # 旋转180度
            transformed_mask = np.rot90(transformed_mask, k=2)
        elif op == 'rot90_3': # 旋转270度
            transformed_mask = np.rot90(transformed_mask, k=3)
        elif op == 'fliplr': # 左右翻转
            transformed_mask = np.fliplr(transformed_mask)
        elif op == 'flipud': # 上下翻转
            transformed_mask = np.flipud(transformed_mask)
        else:
            print(f"警告: 未知的方向变换操作 '{op}'")
    return transformed_mask

def load_multi_organ_segmentation_mask(
    nii_data_array,
    slice_index,
    organ_map_nii,
    organ_channel_map,
    num_organs,
    target_size, # 目标尺寸暂时保留，但我们先在原始尺寸上做方向调整
    nii_path_for_error_msg="",
    orientation_transform_ops=None # 新增参数，例如 ['rot90'] 或 ['rot90', 'fliplr']
):
    """
    从已加载的 NII 数据数组中提取特定切片的分割掩码, 创建多通道掩码。
    根据提供的映射处理标签 (合并左右肾到'kidney', 标签5到'bowel')。
    返回多通道二值掩码(0或1), 形状为(target_size, target_size, num_organs)
    """
    try:
        seg_data = nii_data_array

        if not isinstance(seg_data, np.ndarray) or seg_data.ndim != 3:
             print(f"警告: 传入的NII数据不是有效的3D NumPy数组: {nii_path_for_error_msg}, 形状或类型: {seg_data.shape if isinstance(seg_data, np.ndarray) else type(seg_data)}")
             return None

        num_slices_nii = seg_data.shape[2]
        if not (0 <= slice_index < num_slices_nii):
            # print(f"警告: 切片索引 {slice_index} 超出范围 [0, {num_slices_nii-1}) for {nii_path_for_error_msg}")
            return None

        mask_slice_float = seg_data[:, :, slice_index]
        # --- 关键修改点：应用方向变换 ---
        if orientation_transform_ops:
            print(f"对掩码切片应用方向变换: {orientation_transform_ops}")
            mask_slice_float = apply_orientation_transform(mask_slice_float, orientation_transform_ops)
        # --- 方向变换结束 ---
        
        mask_slice_int = np.round(mask_slice_float).astype(np.int16)

        # 创建多通道掩码 (在变换后的掩码尺寸上创建)
        # 注意：这里 multi_channel_mask 的尺寸是变换后的原始掩码尺寸，还未resize
        multi_channel_mask = np.zeros((mask_slice_int.shape[0], mask_slice_int.shape[1], num_organs), dtype=np.float32)

        for nii_value, organ_name in organ_map_nii.items():
            if organ_name in organ_channel_map:
                channel_idx = organ_channel_map[organ_name]
                if organ_name == 'kidney':
                    binary_mask_organ = ((mask_slice_int == 3) | (mask_slice_int == 4)).astype(np.float32)
                else:
                    binary_mask_organ = (mask_slice_int == nii_value).astype(np.float32)
                
                # 确保 binary_mask_organ 和 multi_channel_mask 的前两维匹配
                if binary_mask_organ.shape == multi_channel_mask.shape[:2]:
                    multi_channel_mask[:, :, channel_idx] += binary_mask_organ
                else:
                    # 如果因为旋转导致尺寸不一致（应该不会，因为旋转保持尺寸），这里需要处理或报错
                    print(f"警告: 器官 {organ_name} 的二值掩码形状 {binary_mask_organ.shape} 与多通道掩码基底形状 {multi_channel_mask.shape[:2]} 不匹配。")
                    # 尝试resize binary_mask_organ 到 multi_channel_mask 的尺寸
                    resized_binary_mask_organ = cv2.resize(binary_mask_organ, (multi_channel_mask.shape[1], multi_channel_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    multi_channel_mask[:, :, channel_idx] += resized_binary_mask_organ


        # --- Resize 操作 ---
        # 现在 multi_channel_mask 是在（可能旋转/翻转过的）原始切片分辨率下的
        # 将其resize到目标尺寸
        if multi_channel_mask.shape[0] != target_size or multi_channel_mask.shape[1] != target_size:
            resized_mask = cv2.resize(
                multi_channel_mask,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST # 对掩码使用最近邻插值
            )
            # cv2.resize可能压缩单通道输出，需要重新扩展维度
            if len(resized_mask.shape) == 2 and num_organs == 1:
                 resized_mask = np.expand_dims(resized_mask, axis=-1)
            elif len(resized_mask.shape) == 2 and num_organs > 1:
                # 这种情况不应该发生，如果发生了，说明resize逻辑有问题，或者输入掩码有问题
                print(f"警告: Resize后多通道掩码被意外压缩为2D: {resized_mask.shape}, 目标通道: {num_organs} from {nii_path_for_error_msg}")
                # 为避免错误，返回None或全零掩码
                return np.zeros((target_size, target_size, num_organs), dtype=np.float32) # 返回全零
            resized_mask = (resized_mask > 0.5).astype(np.float32) # 二值化确保是0或1
        else:
            # 如果原始尺寸（经过变换后）已经是目标尺寸，则直接二值化
            resized_mask = (multi_channel_mask > 0.5).astype(np.float32)

        if resized_mask.shape != (target_size, target_size, num_organs):
             print(f"警告: 最终掩码形状不正确: {resized_mask.shape}，预期: {(target_size, target_size, num_organs)} from {nii_path_for_error_msg}")
             # 可以选择返回None或一个全零的掩码
             return np.zeros((target_size, target_size, num_organs), dtype=np.float32) # 返回全零

        return resized_mask

    except Exception as e:
        print(f"处理分割掩码错误 (来自预加载数据) {nii_path_for_error_msg}, 切片 {slice_index}, 变换 {orientation_transform_ops}: {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈
        return None

def preprocess_image_for_unet(image, target_size):
    """准备单个图像切片作为U-Net输入"""
    # 调整图像大小
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # 扩展到3个通道 (对于需要3通道输入的模型如EfficientNet)
    image_rgb = np.stack([image_resized] * N_INPUT_CHANNELS, axis=-1)
    return image_rgb.astype(np.float32)

def get_dicom_files_dict(patient_id, series_id, dicom_tags_df):
    """辅助函数：获取并排序某个序列的DICOM文件信息"""
    sorted_dicom_info = []
    patient_dir = os.path.join(TRAIN_IMAGES_DIR, str(patient_id))
    series_folder = os.path.join(patient_dir, str(series_id))
    dicom_files = glob.glob(os.path.join(series_folder, '*.dcm'))
    if not dicom_files: return []

    # --- 尝试使用 InstanceNumber 排序 ---
    dicom_tuples = []
    use_tags = False
    # 检查 DICOM tags 是否包含必要信息
    if dicom_tags_df is not None and all(col in dicom_tags_df.columns for col in ['PatientID', 'SeriesInstanceUID', 'InstanceNumber', 'SOPInstanceUID']):
        try:
            # 确保类型匹配
            patient_id_str = str(patient_id)
            
            # 从tags_df获取此序列的信息
            tags_subset = dicom_tags_df[
                 (dicom_tags_df['PatientID'].astype(str) == patient_id_str) &
                 (dicom_tags_df['series_id_extracted'] == str(series_id))
            ][['InstanceNumber', 'SOPInstanceUID']].dropna()

            if not tags_subset.empty:
                sop_to_inst = dict(zip(tags_subset['SOPInstanceUID'], tags_subset['InstanceNumber'].astype(int)))
                use_tags = True # 标记成功使用tags

                for f_path in dicom_files:
                    try:
                        ds_sop = pydicom.dcmread(f_path, stop_before_pixels=True).SOPInstanceUID
                        if ds_sop in sop_to_inst:
                             dicom_tuples.append((sop_to_inst[ds_sop], f_path))
                        else: # Fallback: read InstanceNumber directly from header
                           ds_num = pydicom.dcmread(f_path, stop_before_pixels=True)
                           dicom_tuples.append((int(ds_num.InstanceNumber), f_path))
                    except: # 文件读取失败或其他异常
                         pass # 跳过无法处理的文件

        except Exception as e:
            use_tags = False # 出错则回退

    # --- 如果 Tags 排序失败或不可用，尝试直接从DICOM头读取 InstanceNumber ---
    if not use_tags or not dicom_tuples:
        dicom_tuples = []
        for f_path in dicom_files:
            try:
                ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                dicom_tuples.append((int(ds.InstanceNumber), f_path))
            except Exception:
                pass # 跳过无法读取的文件

    # --- 如果 DICOM 头读取也失败，则按文件名排序 ---
    if not dicom_tuples:
        try:
           # 尝试按文件名中的数字排序
           dicom_tuples = sorted([(int(os.path.splitext(os.path.basename(f))[0]), f) for f in dicom_files])
        except ValueError:
           # 如果文件名不是纯数字，则按字母顺序排序
           dicom_tuples = sorted([(i, f) for i, f in enumerate(sorted(dicom_files))])

    # 按 InstanceNumber (或其他排序键) 排序
    dicom_tuples.sort(key=lambda x: x[0])
    sorted_dicom_info = [(item[0], item[1]) for item in dicom_tuples] # 返回 (InstanceNumber, path)

    return sorted_dicom_info


# 建议在 In[4] visualize_preprocessed_samples 函数的上方或一个新的Cell中添加此函数

def debug_dicom_nifti_alignment(
    dicom_path,
    nii_data_array, # 预加载的整个3D NII数据
    nii_slice_idx,  # 要从NII数据中提取的切片索引
    organ_map_nii,
    organ_channel_map,
    num_organs,
    target_size, # 这是最终模型期望的尺寸
    orientation_transform_ops_list=None # 一个包含多种变换操作列表的列表, e.g., [None, ['rot90'], ['fliplr']]
):
    """
    调试单个DICOM图像和其对应的NIFTI掩码（应用不同方向变换）的对齐情况。
    """
    print(f"调试对齐: DICOM='{os.path.basename(dicom_path)}', NII切片索引={nii_slice_idx}")

    # 1. 加载和预处理DICOM图像
    dicom_image_raw, instance_number = load_dicom_slice(dicom_path)
    if dicom_image_raw is None:
        print(f"无法加载DICOM图像: {dicom_path}")
        return
    
    # 将DICOM图像调整到target_size以进行比较（注意：U-Net输入是3通道的）
    # 为了可视化，我们先用原始单通道灰度图
    dicom_display = cv2.resize(dicom_image_raw, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    if orientation_transform_ops_list is None:
        orientation_transform_ops_list = [None] # 默认只显示原始（无变换）

    num_transforms = len(orientation_transform_ops_list)
    
    # 为每个变换创建一个图
    for i, current_ops in enumerate(orientation_transform_ops_list):
        print(f"\n尝试变换: {current_ops}")
        
        # 2. 加载和处理NIFTI掩码（应用当前方向变换）
        # 注意：这里调用修改后的 load_multi_organ_segmentation_mask
        # 它会在内部处理方向，然后resize到target_size
        nifti_mask_multichannel = load_multi_organ_segmentation_mask(
            nii_data_array,
            nii_slice_idx,
            organ_map_nii,
            organ_channel_map,
            num_organs,
            target_size, # 确保掩码也被resize到同样大小
            nii_path_for_error_msg=f"debug_patient_slice_{nii_slice_idx}",
            orientation_transform_ops=current_ops
        )

        if nifti_mask_multichannel is None:
            print(f"无法为变换 {current_ops} 加载NIFTI掩码。")
            # 可以选择画一个空白掩码图
            fig_title_suffix = f"(变换: {current_ops}) - 掩码加载失败"
            nifti_mask_multichannel_display = np.zeros((target_size, target_size, 3), dtype=np.uint8) # 用于显示的空白彩色图
            blended_display = (dicom_display * 255).astype(np.uint8)
            if len(blended_display.shape) == 2: # 如果是单通道灰度图，转为BGR
                blended_display = cv2.cvtColor(blended_display, cv2.COLOR_GRAY2BGR)

        else:
            fig_title_suffix = f"(变换: {current_ops if current_ops else '无'})"
            # 创建彩色叠加图进行可视化 (与您 visualize_preprocessed_samples 中的逻辑类似)
            colors = {
                'liver': [255, 0, 0],  # 红色 (注意这里用BGR顺序，因为OpenCV常用BGR)
                'spleen': [0, 255, 0],  # 绿色
                'kidney': [0, 0, 255],  # 蓝色
                'bowel': [255, 255, 0]    # 黄色
            }
            
            # 将单通道DICOM显示图像转换为BGR，以便与彩色掩码叠加
            dicom_display_bgr = (dicom_display * 255).astype(np.uint8)
            if len(dicom_display_bgr.shape) == 2:
                dicom_display_bgr = cv2.cvtColor(dicom_display_bgr, cv2.COLOR_GRAY2BGR)

            # 创建掩码的彩色叠加版本
            overlay_mask_display = np.zeros_like(dicom_display_bgr, dtype=np.uint8) # BGR
            for organ_name_map, channel_idx_map in organ_channel_map.items():
                if organ_name_map in colors:
                    color_bgr = colors[organ_name_map] # 直接使用BGR
                    # nifti_mask_multichannel 是 (target_size, target_size, num_organs)
                    organ_mask_slice = nifti_mask_multichannel[:, :, channel_idx_map]
                    # 将单通道二值掩码应用颜色，并叠加到 overlay_mask_display
                    for c in range(3): # B, G, R
                        overlay_mask_display[organ_mask_slice > 0, c] = color_bgr[c]
            
            # 混合图像和掩码
            alpha = 0.4
            blended_display = cv2.addWeighted(dicom_display_bgr, 1 - alpha, overlay_mask_display, alpha, 0)


        # 3. 可视化
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(blended_display, cv2.COLOR_BGR2RGB)) # Matplotlib期望RGB
        plt.title(f"DICOM与NIFTI掩码叠加 {fig_title_suffix}\nDICOM: {os.path.basename(dicom_path)}, NII切片: {nii_slice_idx}")
        plt.axis('off')
        
        # 添加图例 (可选，但推荐)
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=[c/255. for c in colors[org][::-1]], label=org) #转RGB给matplotlib
                           for org in organ_channel_map.keys() if org in colors]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
        plt.show()


def visualize_preprocessed_samples(preprocessed_dir, num_patients=3, samples_per_patient=2):
    """
    从预处理数据中可视化几个样本，检查图像和掩码是否对齐
    
    参数:
        preprocessed_dir: 预处理数据目录
        num_patients: 要检查的患者数量
        samples_per_patient: 每个患者检查的样本数量
    """
    print(f"检查预处理数据的对齐情况...")
    
    # 获取所有预处理过的患者ID
    patient_dirs = [d for d in os.listdir(preprocessed_dir) 
                   if os.path.isdir(os.path.join(preprocessed_dir, d))]
    
    if not patient_dirs:
        print("没有找到预处理数据目录")
        return
    
    # 随机选择几个患者
    selected_patients = np.random.choice(patient_dirs, 
                                        min(num_patients, len(patient_dirs)), 
                                        replace=False)
    
    for patient_id in selected_patients:
        patient_dir = os.path.join(preprocessed_dir, patient_id)
        npz_files = glob.glob(os.path.join(patient_dir, "*.npz"))
        
        if not npz_files:
            print(f"患者 {patient_id} 没有预处理文件")
            continue
        
        print(f"检查患者 {patient_id} 的预处理数据")
        
        # 随机选择几个样本
        selected_files = np.random.choice(npz_files, 
                                         min(samples_per_patient, len(npz_files)), 
                                         replace=False)
        
        for file_path in selected_files:
            try:
                # 加载NPZ文件
                data = np.load(file_path)
                image = data['image']
                mask = data['mask']
                
                # 获取文件名作为切片标识
                slice_id = os.path.splitext(os.path.basename(file_path))[0]
                
                # 创建彩色掩码叠加
                colors = {
                    'liver': [1.0, 0.0, 0.0],  # 红色
                    'spleen': [0.0, 1.0, 0.0],  # 绿色
                    'kidney': [0.0, 0.0, 1.0],  # 蓝色
                    'bowel': [1.0, 1.0, 0.0]    # 黄色
                }
                
                # 准备可视化
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 显示原始图像
                if image.shape[2] == 3:  # 如果是3通道图像
                    axes[0].imshow(image)
                else:  # 如果是单通道图像
                    axes[0].imshow(image[:,:,0], cmap='gray')
                axes[0].set_title(f"原始图像 - 患者 {patient_id}, 切片 {slice_id}")
                axes[0].axis('off')
                
                # 显示多通道掩码 (各通道不同颜色)
                overlay_mask = np.zeros((*mask.shape[0:2], 3))
                for i, organ_name in enumerate(ORGAN_CHANNEL_MAP.keys()):
                    if i < mask.shape[2]:  # 确保通道索引有效
                        color = colors[organ_name]
                        for c in range(3):  # 对RGB三个通道
                            overlay_mask[:,:,c] += mask[:,:,i] * color[c]
                
                # 将掩码值限制在[0,1]范围内
                overlay_mask = np.clip(overlay_mask, 0, 1)
                
                # 显示掩码
                axes[1].imshow(overlay_mask)
                axes[1].set_title("分割掩码")
                axes[1].axis('off')
                
                # 显示图像和掩码叠加
                # 将单通道图像转为RGB
                if image.shape[2] == 3:
                    rgb_image = image
                else:
                    rgb_image = np.stack([image[:,:,0]] * 3, axis=-1)
                
                # 叠加图像
                alpha = 0.5
                blended = rgb_image * (1 - alpha) + overlay_mask * alpha
                blended = np.clip(blended, 0, 1)
                
                axes[2].imshow(blended)
                axes[2].set_title("图像+掩码叠加")
                axes[2].axis('off')
                
                # 添加图例
                legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=colors[organ], label=organ)
                                  for organ in ORGAN_CHANNEL_MAP.keys()]
                fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                plt.show()
                
            except Exception as e:
                print(f"可视化文件 {file_path} 时出错: {e}")
    
    print("预处理数据对齐检查完成")



def preprocess_and_save_data(patient_id, image_paths, nii_path, output_dir, target_size):
    """
    预处理并保存患者的图像和掩码数据
    
    参数:
        patient_id: 患者ID
        image_paths: 该患者的DICOM路径列表
        nii_path: 该患者的NII文件路径
        output_dir: 输出目录
        target_size: 目标图像大小
    
    返回:
        处理成功的切片数量
    """
    patient_output_dir = os.path.join(output_dir, str(patient_id))
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # 加载NII数据
    try:
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata()
    except Exception as e:
        print(f"无法加载患者 {patient_id} 的NII文件: {e}")
        return 0
    
    processed_count = 0
    
    # 处理每个切片
    for slice_idx, dicom_path in enumerate(image_paths):
        if slice_idx >= nii_data.shape[2]:  # 确保不超出NII数据的切片范围
            continue
            
        # 加载DICOM图像
        image, instance_number = load_dicom_slice(dicom_path)
        if image is None:
            continue
            
        # 获取掩码
        mask = load_multi_organ_segmentation_mask(
            nii_data, slice_idx, 
            ORGAN_MAP_NII, ORGAN_CHANNEL_MAP, NUM_ORGANS, 
            target_size, nii_path
        )
        if mask is None:
            continue
            
        # 预处理图像
        processed_image = preprocess_image_for_unet(image, target_size)
        
        # 保存处理后的数据
        output_path = os.path.join(patient_output_dir, f"{instance_number if instance_number else slice_idx}.npz")
        np.savez_compressed(
            output_path,
            image=processed_image,
            mask=mask
        )
        
        processed_count += 1
    
    return processed_count

def perform_offline_preprocessing(image_paths_dict, segmentation_map, output_dir, target_size):
    """
    对所有患者数据进行离线预处理
    
    参数:
        image_paths_dict: 患者ID到DICOM路径列表的映射
        segmentation_map: 患者ID到NII路径的映射
        output_dir: 输出目录
        target_size: 目标图像大小
    
    返回:
        预处理数据的患者ID列表
    """
    print("开始离线预处理数据...")
    
    # 获取所有需要处理的患者ID
    patient_ids = sorted(list(set(image_paths_dict.keys()) & set(segmentation_map.keys())))
    
    if not patient_ids:
        print("没有找到同时包含图像和分割数据的患者")
        return []
    
    print(f"将对 {len(patient_ids)} 位患者的数据进行预处理")
    
    # 使用多进程处理
    total_processed = 0
    preprocessed_patients = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_patient = {
            executor.submit(
                preprocess_and_save_data,
                patient_id,
                image_paths_dict[patient_id],
                segmentation_map[patient_id],
                output_dir,
                target_size
            ): patient_id for patient_id in patient_ids
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_patient), total=len(patient_ids), desc="预处理患者数据"):
            patient_id = future_to_patient[future]
            try:
                processed_count = future.result()
                if processed_count > 0:
                    total_processed += processed_count
                    preprocessed_patients.append(patient_id)
                    print(f"患者 {patient_id} 预处理完成: {processed_count} 个切片")
                else:
                    print(f"患者 {patient_id} 没有处理成功的切片")
            except Exception as e:
                print(f"处理患者 {patient_id} 时出错: {e}")
    
    print(f"预处理完成，共处理 {len(preprocessed_patients)}/{len(patient_ids)} 位患者的 {total_processed} 个切片")
    return preprocessed_patients



# 内存高效版 - 使用生成器而不是预加载
def create_tf_dataset_from_preprocessed(patient_ids, preprocessed_dir, batch_size, augment=True, shuffle=True):
    """
    从预处理数据创建tf.data.Dataset，使用生成器方式避免内存溢出
    
    参数:
        patient_ids: 患者ID列表
        preprocessed_dir: 预处理数据目录
        batch_size: 批量大小
        augment: 是否进行数据增强
        shuffle: 是否打乱数据
    
    返回:
        tf.data.Dataset对象
    """
    # 收集所有预处理文件的路径
    all_files = []
    for patient_id in patient_ids:
        patient_dir = os.path.join(preprocessed_dir, str(patient_id))
        if not os.path.exists(patient_dir):
            continue
        
        npz_files = glob.glob(os.path.join(patient_dir, "*.npz"))
        all_files.extend(npz_files)
    
    if not all_files:
        raise ValueError(f"没有找到预处理数据文件，请先运行预处理")
    
    print(f"找到 {len(all_files)} 个预处理数据文件")
    
    # 创建一个基于文件路径的数据集
    paths_dataset = tf.data.Dataset.from_tensor_slices(all_files)
    
    if shuffle:
        # 限制缓冲区大小，避免内存问题
        buffer_size = min(len(all_files), 10000)
        paths_dataset = paths_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # 定义加载函数
    def load_npz_file(file_path):
        """加载单个NPZ文件"""
        # 将张量转换为字符串
        file_path_str = file_path.numpy().decode('utf-8')
        
        try:
            data = np.load(file_path_str)
            image = data['image'].astype(np.float32)
            mask = data['mask'].astype(np.float32)
            
            # 确保形状正确
            if image.shape != (TARGET_SIZE, TARGET_SIZE, N_INPUT_CHANNELS) or mask.shape != (TARGET_SIZE, TARGET_SIZE, NUM_ORGANS):
                print(f"警告: 文件 {file_path_str} 的形状不正确，图像: {image.shape}, 掩码: {mask.shape}")
                # 返回正确形状的零数组
                return (
                    np.zeros((TARGET_SIZE, TARGET_SIZE, N_INPUT_CHANNELS), dtype=np.float32),
                    np.zeros((TARGET_SIZE, TARGET_SIZE, NUM_ORGANS), dtype=np.float32)
                )
                
            return image, mask
            
        except Exception as e:
            print(f"加载文件 {file_path_str} 失败: {e}")
            # 返回零数组
            return (
                np.zeros((TARGET_SIZE, TARGET_SIZE, N_INPUT_CHANNELS), dtype=np.float32),
                np.zeros((TARGET_SIZE, TARGET_SIZE, NUM_ORGANS), dtype=np.float32)
            )
    
    # 使用py_function将路径映射到图像和掩码
    def load_and_process(file_path):
        image, mask = tf.py_function(
            load_npz_file,
            [file_path],
            [tf.float32, tf.float32]
        )
        # 设置形状，避免形状推断问题
        image.set_shape((TARGET_SIZE, TARGET_SIZE, N_INPUT_CHANNELS))
        mask.set_shape((TARGET_SIZE, TARGET_SIZE, NUM_ORGANS))
        return image, mask
    
    # 映射加载函数
    dataset = paths_dataset.map(load_and_process, num_parallel_calls=PARALLEL_CALLS)
    
    # 过滤掉加载失败的文件（可选）
    # dataset = dataset.filter(lambda img, mask: tf.reduce_sum(img) > 0)
    
    # 数据增强
    def augment_data(image, mask):
        # 随机水平翻转
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # 随机亮度
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_brightness(image, max_delta=0.1)
            # 确保值在[0,1]范围内
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        # 随机对比度
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    
    # 应用数据增强
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=PARALLEL_CALLS)
    
    # 批处理和预取
    dataset = dataset.batch(batch_size)
    
    # 对于大型数据集，最好不要缓存
    if CACHE_DATASET:
        print("警告: 对大型数据集启用缓存可能导致内存问题，考虑设置CACHE_DATASET=False")
        dataset = dataset.cache()
    
    return dataset.prefetch(PREFETCH_BUFFER_SIZE)



# 数据分析函数 - 完整版
def verify_nii_labels(segmentation_map, expected_map):
    """验证NII文件中的标签值是否与预期的器官映射匹配"""
    print("开始验证NII文件中的标签值...")
    # 定义原始NII文件中的预期非背景标签值
    expected_nii_labels_set = {1, 2, 3, 4, 5} # 肝、脾、左肾、右肾、肠

    found_labels = set()
    label_counts = {}
    label_pixels = {val: [] for val in range(6)} # 统计0-5

    sample_patients = list(segmentation_map.keys())[:20]
    print(f"将抽样检查 {len(sample_patients)} 位患者的NII文件...")

    for patient_id in tqdm(sample_patients, desc="验证NII标签"):
        nii_path = segmentation_map.get(patient_id)
        if not nii_path: continue
        try:
            nii_img = nib.load(nii_path)
            # 使用默认的 float dtype 加载数据，避免类型错误
            seg_data = nii_img.get_fdata() 

            unique_labels_in_file = np.unique(seg_data)
            found_labels.update(unique_labels_in_file)

            for label in unique_labels_in_file:
                 # 转换为整数进行比较和字典键查找
                 label_int = int(np.round(label)) # 四舍五入并转整数，处理可能的浮点误差
                 if 0 <= label_int <= 5: # 只统计0-5的标签
                    # 使用原始浮点标签值进行精确计数
                    count = np.sum(np.round(seg_data) == label_int) # 使用四舍五入比较
                    if count > 0:
                       label_pixels[label_int].append(count)

        except Exception as e:
            print(f"处理患者 {patient_id} 的NII文件时出错: {e}")

    print("\n=== NII文件标签验证结果 ===")
    # 从找到的标签中提取整数标签
    found_int_labels = {int(np.round(l)) for l in found_labels if l == np.round(l)}
    print(f"在抽样的NII文件中发现的所有整数标签值: {sorted(list(found_int_labels))}")


    found_numeric_labels = {int(l) for l in found_int_labels if l != 0}

    missing_labels = expected_nii_labels_set - found_numeric_labels
    extra_labels = found_numeric_labels - expected_nii_labels_set

    if missing_labels:
        print(f"警告: 以下预期标签在抽样NII文件中未找到: {missing_labels}")
    else:
        print("所有预期的器官标签 (1-5) 至少在部分抽样文件中存在。")

    if extra_labels:
        print(f"警告: NII文件中包含以下预期之外的整数标签: {extra_labels}")
    else:
        print("未发现预期之外的整数标签。")

    # 打印像素统计
    print("\n各标签值的像素数量统计 (基于抽样文件):")
    name_map = {1: 'liver', 2: 'spleen', 3: 'kidney_left', 4: 'kidney_right', 5: 'bowel', 0: 'background'}
    for label_int, counts in label_pixels.items():
        if not counts: continue
        organ_name = name_map.get(label_int, "未知")
        avg_count = np.mean(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        print(f"  标签 {label_int} ({organ_name}): 平均像素数 = {avg_count:.0f}, 最小值 = {min_count:.0f}, 最大值 = {max_count:.0f}, 样本数 = {len(counts)}")

    return found_labels # 返回原始找到的标签（可能包含浮点数）

def analyze_organ_distribution(segmentation_map, organ_map_nii_model):
    """分析各器官（按模型定义合并）在数据集中的分布情况"""
    print("开始分析器官分布...")

    organ_slice_counts = {organ: 0 for organ in organ_map_nii_model.values()}
    organ_pixel_counts = {organ: 0 for organ in organ_map_nii_model.values()}
    total_slices = 0
    total_patients = len(segmentation_map)
    patients_with_organ = {organ: set() for organ in organ_map_nii_model.values()}

    for patient_id, nii_path in tqdm(segmentation_map.items(), desc="分析器官分布"):
        try:
            nii_img = nib.load(nii_path)
            # 使用默认的 float dtype 加载数据
            seg_data = nii_img.get_fdata() 
            # 四舍五入为整数以进行标签比较
            seg_data_int = np.round(seg_data).astype(np.int16)

            num_slices_in_scan = seg_data_int.shape[2]
            total_slices += num_slices_in_scan

            for slice_idx in range(num_slices_in_scan):
                slice_data_int = seg_data_int[:, :, slice_idx]

                # 检查每个模型定义的器官是否存在于切片中
                for nii_value, organ_name in organ_map_nii_model.items():
                    if organ_name == 'kidney': # 合并处理肾脏
                        has_organ = np.any((slice_data_int == 3) | (slice_data_int == 4))
                        pixel_count = np.sum((slice_data_int == 3) | (slice_data_int == 4))
                    else: # 处理其他器官 (肝脏 1, 脾脏 2, 肠道 5)
                        has_organ = np.any(slice_data_int == nii_value)
                        pixel_count = np.sum(slice_data_int == nii_value)

                    if has_organ:
                        organ_slice_counts[organ_name] += 1
                        organ_pixel_counts[organ_name] += pixel_count
                        patients_with_organ[organ_name].add(patient_id)

        except Exception as e:
            print(f"分析患者 {patient_id} 时出错: {e}")

    organ_patient_counts = {organ: len(pids) for organ, pids in patients_with_organ.items()}

    print("\n=== 器官分布分析结果 ===")
    print(f"总患者数: {total_patients}")
    print(f"总切片数 (所有NII文件): {total_slices}")

    print("\n器官在患者中的分布:")
    for organ, count in organ_patient_counts.items():
        percentage = count / total_patients * 100 if total_patients > 0 else 0
        print(f"  {organ}: {count}/{total_patients} 患者 ({percentage:.2f}%)")

    print("\n器官在切片中的分布 (至少有一个像素):")
    for organ, count in organ_slice_counts.items():
        percentage = count / total_slices * 100 if total_slices > 0 else 0
        print(f"  {organ}: {count}/{total_slices} 切片 ({percentage:.2f}%)")

    print("\n器官总像素数量:")
    for organ, count in organ_pixel_counts.items():
        avg_per_slice_present = count / max(organ_slice_counts[organ], 1)
        print(f"  {organ}: 总像素数 = {count}, 平均每(含器官)切片像素数 = {avg_per_slice_present:.2f}")

    # === 计算类别权重 (基于切片频率倒数) ===
    class_weights_slice_inv = {}
    if total_slices > 0: # 使用总切片数作为分母计算频率
        max_slice_count = max(organ_slice_counts.values()) if organ_slice_counts else 1.0
        # 另一种方法：权重与频率成反比，再归一化
        for organ, count in organ_slice_counts.items():
             # 频率 = 该器官出现切片数 / 总切片数
             frequency = (count + 1e-6) / total_slices
             # 权重与频率成反比，用最大计数的倒数比例
             weight = max_slice_count / (count + 1e-6)
             class_weights_slice_inv[organ] = weight

        # 归一化 (例如，使最小权重为1)
        min_weight = min(class_weights_slice_inv.values()) if class_weights_slice_inv else 1.0
        if min_weight > 0:
             for organ in class_weights_slice_inv:
                  class_weights_slice_inv[organ] /= min_weight
        else: # 如果有器官从未出现，权重可能无限大，需要处理
             max_finite_weight = max([w for w in class_weights_slice_inv.values() if np.isfinite(w)], default=1.0)
             for organ in class_weights_slice_inv:
                  if not np.isfinite(class_weights_slice_inv[organ]):
                       class_weights_slice_inv[organ] = max_finite_weight * 2 # 给一个较大的有限值
             min_weight = min(class_weights_slice_inv.values())
             if min_weight > 0:
                for organ in class_weights_slice_inv:
                     class_weights_slice_inv[organ] /= min_weight
    else: # 如果没有有效的切片
        class_weights_slice_inv = {organ: 1.0 for organ in organ_map_nii_model.values()}

    print("\n建议的类别权重 (基于切片频率倒数，归一化):")
    for organ, weight in class_weights_slice_inv.items():
        print(f"  {organ}: {weight:.2f}")

    return {
        'organ_patient_counts': organ_patient_counts,
        'organ_slice_counts': organ_slice_counts,
        'organ_pixel_counts': organ_pixel_counts,
        'class_weights': class_weights_slice_inv
    }



# === U-Net模型定义 ===
def build_unet_multi_organ(input_shape, num_organs, dropout_rate=0.3):
    """构建带有EfficientNetB0编码器的2D U-Net模型用于多器官分割"""
    print(f"构建多器官U-Net (EfficientNetB0 编码器), 输入形状:{input_shape}, 输出通道数:{num_organs}")

    # 加载预训练的EfficientNetB0作为编码器
    efficientnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

    # 获取跳跃连接层 (使用名称获取更稳定)
    try:
        # 这些层名通常在EfficientNet B0-B7中比较稳定
        s1 = efficientnet.get_layer('block2a_expand_activation').output # 112x112
        s2 = efficientnet.get_layer('block3a_expand_activation').output # 56x56
        s3 = efficientnet.get_layer('block4a_expand_activation').output # 28x28
        s4 = efficientnet.get_layer('block6a_expand_activation').output # 14x14
        b0 = efficientnet.output # 7x7 (瓶颈)
        skip_connections = [s1, s2, s3, s4]
        print("成功获取EfficientNetB0中间层作为跳跃连接。")
    except ValueError as e:
        print(f"错误：无法获取指定的EfficientNetB0层。请检查层名称: {e}")
        print("建议使用 model.summary() 检查实际层名并更新。")
        # print(efficientnet.summary()) # 打印模型结构以帮助调试
        raise e

    # === 解码器 ===
    # 定义上采样/解码器块的滤波器数量
    decoder_filters = [256, 128, 64, 32] # 从瓶颈向上
    x = b0

    # 解码器路径与跳跃连接
    for i in range(len(decoder_filters)):
        filters = decoder_filters[i]
        # 获取对应的跳跃连接 (从深到浅)
        skip = skip_connections[len(skip_connections) - 1 - i]

        # 上采样 (Conv2DTranspose)
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)

        # 检查并调整尺寸以匹配跳跃连接 (如果需要)
        # if x.shape[1:3] != skip.shape[1:3]:
        #     print(f"尺寸不匹配: 上采样后 {x.shape[1:3]}, 跳跃连接 {skip.shape[1:3]}. 调整解码器输出大小。")
        #     x = tf.image.resize(x, skip.shape[1:3], method='bilinear')
        #   或者调整跳跃连接 (有时更简单):
        #   skip = layers.Conv2D(filters, 1, padding='same', activation='relu')(skip) # 用1x1卷积调整通道数
        #   skip = tf.image.resize(skip, x.shape[1:3], method='bilinear')

        # 连接跳跃特征
        x = layers.concatenate([x, skip], axis=-1)

        # 两个卷积层 + ReLU + Dropout
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x) # 添加BN层有助于稳定训练
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    # 最终上采样到原始输入大小 (224x224)
    # 当前 x 的大小应为 112x112 (经过4次上采样)
    # 再进行一次上采样
    x = layers.Conv2DTranspose(16, (2, 2), strides=2, padding='same', activation='relu')(x) # 输出 224x224x16
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # 输出层: 1x1 卷积，通道数为器官数，激活函数为 sigmoid (用于多标签分割)
    outputs = layers.Conv2D(num_organs, 1, activation='sigmoid', name='multi_organ_mask')(x)

    # 创建模型
    model = models.Model(inputs=efficientnet.input, outputs=outputs, name=f"U-Net_EffB0_{num_organs}Organ")

    return model


# 在 Cell 9 ("损失函数&评价指标")

SMOOTH = 1e-6

@tf.function
def dice_coefficient(y_true, y_pred):
    """计算单个通道/类别的Dice系数"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + SMOOTH) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + SMOOTH)
    return dice

@tf.function
def dice_loss_single_channel(y_true, y_pred):
    """计算单个通道的 1 - Dice系数 作为损失"""
    return 1.0 - dice_coefficient(y_true, y_pred)

# --- Focal Loss (基于Binary Crossentropy) ---
@tf.function
def focal_loss_bce(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Binary Focal Loss.
    FL(pt) = -alpha_t * (1 - pt)**gamma * log(pt)
    pt is the probability of the true class.
    """
    y_pred = tf.clip_by_value(y_pred, SMOOTH, 1.0 - SMOOTH) # 避免log(0)
    
    # Calculate Chross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    
    # Calculate P_t
    p_t = (y_true * y_pred) + ((1.0 - y_true) * (1.0 - y_pred))
    
    # Calculate Focal Loss
    focal_term = (1.0 - p_t) ** gamma
    
    # Weighted Focal Loss
    loss = alpha * focal_term * cross_entropy # 使用固定的 alpha (可调整)
    
    return tf.reduce_mean(loss) # 对batch和像素取平均


# --- 新的 Focal Dice Loss ---
def create_focal_dice_loss(gamma_focal=2.0, alpha_focal=0.25, lambda_focal=0.5, lambda_dice=0.5, class_weights=None):
    """
    创建结合 Focal Loss (基于BCE) 和 Dice Loss 的损失函数，支持类别权重。
    Args:
        gamma_focal: Focal loss的gamma参数.
        alpha_focal: Focal loss的alpha参数 (单个值，或每个类别的列表/数组).
        lambda_focal: Focal loss的权重.
        lambda_dice: Dice loss的权重.
        class_weights: 每个器官通道的权重列表/数组，用于加权Dice Loss和Focal Loss (如果alpha_focal是单个值).
                       顺序应与 ORGAN_CHANNEL_MAP 中的通道索引一致。
    """
    _class_weights = tf.constant(class_weights if class_weights is not None else [1.0] * NUM_ORGANS, dtype=tf.float32)
    _alpha_focal = alpha_focal # 可以是单个值或列表/数组

    @tf.function
    def focal_dice_loss_fn(y_true, y_pred):
        total_loss = tf.constant(0.0, dtype=tf.float32)
        
        for i in range(NUM_ORGANS):
            y_true_ch = y_true[..., i]
            y_pred_ch = y_pred[..., i]
            
            # Dice Loss for this channel
            dice_l = dice_loss_single_channel(y_true_ch, y_pred_ch)
            
            # Focal Loss (BCE based) for this channel
            # 如果 alpha_focal 是列表，则按通道取值
            current_alpha = _alpha_focal[i] if isinstance(_alpha_focal, (list, tuple, tf.Tensor, np.ndarray)) and len(_alpha_focal) == NUM_ORGANS else _alpha_focal
            focal_l = focal_loss_bce(y_true_ch, y_pred_ch, gamma=gamma_focal, alpha=current_alpha)
            
            # 结合 Focal Loss 和 Dice Loss，并应用类别权重
            channel_loss = (lambda_focal * focal_l + lambda_dice * dice_l) * _class_weights[i]
            total_loss += channel_loss
            
        return total_loss / tf.reduce_sum(_class_weights) # 加权平均或总和，这里用加权平均
        # 或者 return total_loss / tf.cast(NUM_ORGANS, tf.float32) 如果不希望权重影响总损失的尺度

    return focal_dice_loss_fn

# --- 各器官的Dice系数指标 (用于评估 - 保持不变) ---
@tf.function
def dice_liver(y_true, y_pred):
    channel_idx = ORGAN_CHANNEL_MAP['liver']
    return dice_coefficient(y_true[..., channel_idx], y_pred[..., channel_idx])

@tf.function
def dice_spleen(y_true, y_pred):
    channel_idx = ORGAN_CHANNEL_MAP['spleen']
    return dice_coefficient(y_true[..., channel_idx], y_pred[..., channel_idx])

@tf.function
def dice_kidney(y_true, y_pred):
    channel_idx = ORGAN_CHANNEL_MAP['kidney']
    return dice_coefficient(y_true[..., channel_idx], y_pred[..., channel_idx])

@tf.function
def dice_bowel(y_true, y_pred):
    channel_idx = ORGAN_CHANNEL_MAP['bowel']
    return dice_coefficient(y_true[..., channel_idx], y_pred[..., channel_idx])

# 用于训练的评估指标列表
METRICS = [
    #average_dice_coefficient, # 我们将使用新的损失，这个平均的可以去掉，或者保留用于观察
    dice_liver,
    dice_spleen,
    dice_kidney,
    dice_bowel
]


# 在 Cell 10 ("分阶段训练函数")

def train_in_stages(train_pids, val_pids, preprocessed_dir, target_size_int_param,
                      input_channels, num_organs_param, batch_size, 
                      initial_learning_rate, epochs_per_stage, class_weights_map): # class_weights_map 从 analyze_organ_distribution 获取
    
    input_shape = (target_size_int_param, target_size_int_param, input_channels)
    print(f"构建模型，输入形状: {input_shape}, 器官数: {num_organs_param}")
    unet_model = build_unet_multi_organ(input_shape, num_organs_param, dropout_rate=0.3) # 使用您 Cell 8 的定义

    # 准备类别权重，确保顺序与 ORGAN_CHANNEL_MAP 一致
    # 这里的 class_weights_map 应该是类似 {'liver': w1, 'spleen': w2, ...} 的字典
    # 我们需要将其转换为一个列表，顺序与模型输出通道对应
    ordered_class_weights = [1.0] * num_organs_param # 初始化为1.0
    if class_weights_map: # 确保 class_weights_map 不是 None
        for organ, idx in ORGAN_CHANNEL_MAP.items(): # ORGAN_CHANNEL_MAP 是全局的
            if organ in class_weights_map:
                ordered_class_weights[idx] = class_weights_map[organ]
    print(f"训练中使用的有序类别权重: {ordered_class_weights}")

    # ***** 定义新的损失函数实例 *****
    # 您可以调整 FocalDiceLoss 的超参数
    # 例如，给罕见或难分的器官更高的权重（通过 class_weights）
    # lambda_focal 和 lambda_dice 控制两部分损失的贡献，论文没有指明，可以设为0.5, 0.5开始
    final_stage_loss = create_focal_dice_loss(
        gamma_focal=2.0, 
        alpha_focal=0.25, # 或者可以是一个列表，为每个通道设置不同的alpha
        lambda_focal=0.5, 
        lambda_dice=0.5, 
        class_weights=ordered_class_weights
    )
    # ********************************

    # --- 定义各阶段参数 ---
    # 对于前几个阶段，如果只想关注特定器官，可以创建只针对那些器官的损失或权重
    # 例如，Stage1_Liver 可以继续使用 1.0 - dice_liver
    # 或者也使用 FocalDiceLoss，但权重只给 liver
    
    # 为每个阶段动态创建损失函数
    stage_losses = []
    for i in range(len(epochs_per_stage)):
        current_weights = [0.0] * num_organs_param
        if i == 0: # Stage 1: Liver
            current_weights[ORGAN_CHANNEL_MAP['liver']] = ordered_class_weights[ORGAN_CHANNEL_MAP['liver']] \
                                                            if 'liver' in ORGAN_CHANNEL_MAP and ordered_class_weights else 1.0
            stage_losses.append(create_focal_dice_loss(class_weights=current_weights, lambda_focal=0.5, lambda_dice=0.5)) # 可以调整lambda
        elif i == 1: # Stage 2: Liver, Spleen
            if 'liver' in ORGAN_CHANNEL_MAP: current_weights[ORGAN_CHANNEL_MAP['liver']] = ordered_class_weights[ORGAN_CHANNEL_MAP['liver']] if ordered_class_weights else 1.0
            if 'spleen' in ORGAN_CHANNEL_MAP: current_weights[ORGAN_CHANNEL_MAP['spleen']] = ordered_class_weights[ORGAN_CHANNEL_MAP['spleen']] if ordered_class_weights else 1.0
            stage_losses.append(create_focal_dice_loss(class_weights=current_weights, lambda_focal=0.5, lambda_dice=0.5))
        elif i == 2: # Stage 3: Liver, Spleen, Kidney
            if 'liver' in ORGAN_CHANNEL_MAP: current_weights[ORGAN_CHANNEL_MAP['liver']] = ordered_class_weights[ORGAN_CHANNEL_MAP['liver']] if ordered_class_weights else 1.0
            if 'spleen' in ORGAN_CHANNEL_MAP: current_weights[ORGAN_CHANNEL_MAP['spleen']] = ordered_class_weights[ORGAN_CHANNEL_MAP['spleen']] if ordered_class_weights else 1.0
            if 'kidney' in ORGAN_CHANNEL_MAP: current_weights[ORGAN_CHANNEL_MAP['kidney']] = ordered_class_weights[ORGAN_CHANNEL_MAP['kidney']] if ordered_class_weights else 1.0
            stage_losses.append(create_focal_dice_loss(class_weights=current_weights, lambda_focal=0.5, lambda_dice=0.5))
        elif i == 3: # Stage 4: All Organs
            stage_losses.append(final_stage_loss) # 使用为所有器官配置的FocalDiceLoss

    stages = [
        {'name': 'Stage1_LiverFocus', 'epochs': epochs_per_stage[0], 'lr_factor': 1.0, 
         'loss': stage_losses[0], 'monitor': 'val_dice_liver'}, # 仍然监控val_dice_liver
        {'name': 'Stage2_LiverSpleenFocus', 'epochs': epochs_per_stage[1], 'lr_factor': 0.5,
         'loss': stage_losses[1], 'monitor': 'val_average_dice_coefficient'}, # 监控平均Dice
        {'name': 'Stage3_LiverSpleenKidneyFocus', 'epochs': epochs_per_stage[2], 'lr_factor': 0.2,
         'loss': stage_losses[2], 'monitor': 'val_average_dice_coefficient'},
        {'name': 'Stage4_AllOrgans', 'epochs': epochs_per_stage[3], 'lr_factor': 0.1,
         'loss': stage_losses[3], 'monitor': 'val_average_dice_coefficient'} # 最终监控平均Dice
    ]

    stage_histories = {}
    # MODEL_SAVE_PATH 现在应该指向 /kaggle/working/...
    # best_model_path_overall = MODEL_SAVE_PATH # 已在 Cell 2 全局定义并修正

    for i_stage_loop, stage_info in enumerate(stages): # 使用新的索引名避免与外部i_stage冲突
        stage_name = stage_info['name']
        epochs = stage_info['epochs']
        current_lr = initial_learning_rate * stage_info['lr_factor']
        loss_func_for_stage = stage_info['loss'] # 这是已经创建好的损失函数实例
        monitor_metric = stage_info['monitor']
        
        # 确保MODEL_OUTPUT_DIR是全局定义的 /kaggle/working/unet_model_v2
        stage_model_save_path = os.path.join(MODEL_OUTPUT_DIR, f"{stage_name}_best_model.keras")

        print(f"\n=== {stage_name} 训练 ===")
        # ... (打印学习率、监控指标等信息不变) ...
        print(f"  阶段模型将保存到: {stage_model_save_path}")
        if stage_name == stages[-1]['name']: # 检查是否是最后一个阶段
            print(f"  最终最佳模型将保存到: {MODEL_SAVE_PATH}") # 使用全局 MODEL_SAVE_PATH


        train_dataset = create_tf_dataset_from_preprocessed(
            train_pids, preprocessed_dir, batch_size, augment=True, shuffle=True
        )
        val_dataset = create_tf_dataset_from_preprocessed(
            val_pids, preprocessed_dir, batch_size, augment=False, shuffle=False
        )

        optimizer = optimizers.Adam(learning_rate=current_lr)
        
        # 在这里编译模型，使用当前阶段的损失函数
        unet_model.compile(optimizer=optimizer, loss=loss_func_for_stage, metrics=METRICS)
        print(f"模型已为阶段 {stage_name} 编译，损失函数: {loss_func_for_stage.__name__ if hasattr(loss_func_for_stage, '__name__') else str(loss_func_for_stage)}")


        # 回调函数 (与您之前的版本类似，确保路径正确)
        callbacks_list_stage = [
            callbacks.ModelCheckpoint(
                stage_model_save_path, # 保存到阶段特定的路径
                monitor=monitor_metric, mode='max', save_best_only=True,
                save_weights_only=False, verbose=1
            ),
            callbacks.EarlyStopping(
                monitor=monitor_metric, mode='max', patience=EARLY_STOPPING_PATIENCE,
                verbose=1, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, mode='max', factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(OUTPUT_DIR, 'logs', stage_name),
                histogram_freq=1, update_freq='epoch'
            )
        ]
        
        if stage_name == stages[-1]['name']: # 如果是最后一个阶段
            checkpoint_callback_final = callbacks.ModelCheckpoint(
                MODEL_SAVE_PATH, # 全局定义的最终模型保存路径
                monitor=monitor_metric, mode='max', save_best_only=True,
                save_weights_only=False, verbose=1, save_freq='epoch'
            )
            callbacks_list_stage.append(checkpoint_callback_final)
        
        print(f"开始训练阶段: {stage_name}，共 {epochs} 个 Epochs")
        history = unet_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks_list_stage,
            verbose=1
        )
        stage_histories[stage_name] = history.history

        # 在每个阶段结束后，从该阶段的检查点加载最佳模型
        if os.path.exists(stage_model_save_path):
            print(f"阶段 '{stage_name}' 完成。从检查点 '{stage_model_save_path}' 加载此阶段的最佳模型...")
            custom_objects_for_load = { # 只需要自定义指标
                #'average_dice_coefficient': average_dice_coefficient, # 如果在METRICS中使用了
                'dice_liver': dice_liver, 'dice_spleen': dice_spleen,
                'dice_kidney': dice_kidney, 'dice_bowel': dice_bowel
            }
            try:
                # 加载模型时不编译，因为下一阶段会重新编译
                unet_model = models.load_model(stage_model_save_path, custom_objects=custom_objects_for_load, compile=False)
                print(f"模型已从 {stage_model_save_path} 成功加载结构和权重。")
            except Exception as e_load:
                print(f"警告: 从阶段检查点 '{stage_model_save_path}' 加载模型失败: {e_load}")
                print("将继续使用内存中当前的模型（可能已由EarlyStopping恢复了最佳权重）。")
        else:
            print(f"警告: 阶段检查点文件 '{stage_model_save_path}' 未找到。将使用内存中当前阶段训练后的模型。")
        
        gc.collect()

    print("\n所有训练阶段完成。")
    return stage_histories


# 评估函数
# 在 Cell 11 ("模型评估") 中，修改 process_patient_evaluation 函数

# (确保 apply_orientation_transform 函数在此作用域内可用，或者 BEST_NIFTI_ORIENTATION_TRANSFORM 和 USE_REVERSE_NIFTI_MAPPING 是全局的)

def process_patient_evaluation(patient_id, nii_path, dicom_paths, series_pred_dir, pred_files,
                                 organ_map_nii_local, organ_channel_map_local, target_size_local_int, # 使用局部变量名和整数尺寸
                                 prediction_threshold):
    local_dice_scores = {organ: [] for organ in organ_channel_map_local.keys()}
    local_iou_scores = {organ: [] for organ in organ_channel_map_local.keys()}
    local_processed = 0

    try:
        nii_img = nib.load(nii_path)
        gt_data_float = nii_img.get_fdata(dtype=np.float32) # 直接加载为 float32
        nii_total_slices = gt_data_float.shape[2]
    except Exception as e:
        print(f"评估时无法加载患者 {patient_id} 的真实掩码 '{nii_path}': {e}")
        return local_dice_scores, local_iou_scores, local_processed

    # ... (instance_to_pred_path 的逻辑不变) ...
    instance_to_pred_path = {
        int(os.path.splitext(os.path.basename(f))[0]): os.path.join(series_pred_dir, f) # 确保路径完整
        for f in os.listdir(series_pred_dir) # 直接 listdir series_pred_dir
        if os.path.splitext(os.path.basename(f))[0].isdigit() and f.endswith(".npz")
    }
    # 为非数字文件名添加 (如果您的预测文件名可能不是纯数字)
    for f in os.listdir(series_pred_dir):
        if f.endswith(".npz"):
            basename = os.path.splitext(os.path.basename(f))[0]
            if not basename.isdigit():
                instance_to_pred_path[basename] = os.path.join(series_pred_dir, f)


    # dicom_paths 是一个 (instance_number, path) 的元组列表
    # 我们需要的是DICOM在其原始排序列表中的索引 (dicom_list_idx)
    for dicom_list_idx, (instance_number, dicom_path) in enumerate(dicom_paths):
        # instance_number 已经是整数了，来自 get_dicom_files_dict
        if instance_number is None: # 以防万一
            print(f"警告: 患者 {patient_id} 的 DICOM {dicom_path} 缺少InstanceNumber，使用列表索引。")
            # 如果instance_number可能为None，需要一个备用方案来匹配预测文件，
            # 或者在get_dicom_files_dict中确保instance_number总是一个有效值或唯一标识符
            id_for_pred = f"idx{dicom_list_idx}" # 假设预测文件名可能是基于索引的
        else:
            id_for_pred = instance_number


        # ***** 核心修改：获取正确的NIFTI切片索引 *****
        nii_slice_idx_to_use = -1
        if USE_REVERSE_NIFTI_MAPPING: # 使用全局配置
            nii_slice_idx_to_use = nii_total_slices - 1 - dicom_list_idx
        else:
            nii_slice_idx_to_use = dicom_list_idx
        # *********************************************

        if not (0 <= nii_slice_idx_to_use < nii_total_slices):
            # print(f"评估患者 {patient_id}: NIFTI索引 {nii_slice_idx_to_use} (来自DICOM列表索引 {dicom_list_idx}) 超出范围。")
            continue
            
        pred_file_path = instance_to_pred_path.get(id_for_pred)
        if not pred_file_path and str(id_for_pred) in instance_to_pred_path: # 再尝试字符串形式的键
            pred_file_path = instance_to_pred_path[str(id_for_pred)]
            
        if not pred_file_path:
            # print(f"评估患者 {patient_id}: 未找到 InstanceNumber/ID {id_for_pred} 对应的预测文件。")
            continue
            
        try:
            pred_data = np.load(pred_file_path)
            # 假设 'mask' 是 (H, W, NumChannels) 并且是概率值或二值化后的 (0或1)
            pred_mask_from_npz = pred_data['mask'] 
            # 如果保存的是概率，在这里应用阈值；如果已经是二值，确保是float32
            pred_mask_binary = (pred_mask_from_npz > prediction_threshold).astype(np.float32)

            if pred_mask_binary.shape[:2] != (target_size_local_int, target_size_local_int) or \
               pred_mask_binary.shape[2] != len(organ_channel_map_local):
                print(f"评估患者 {patient_id}: 预测掩码 {os.path.basename(pred_file_path)} 形状 {pred_mask_binary.shape} 不正确。预期 ({target_size_local_int},{target_size_local_int},{len(organ_channel_map_local)})")
                continue
        except Exception as e_load_pred:
            print(f"评估患者 {patient_id}: 加载或处理预测掩码 {os.path.basename(pred_file_path)} 失败: {e_load_pred}")
            continue

        gt_slice_raw = gt_data_float[:, :, nii_slice_idx_to_use]

        # ***** 核心修改：对真实掩码应用方向变换 *****
        gt_slice_oriented = apply_orientation_transform(gt_slice_raw, BEST_NIFTI_ORIENTATION_TRANSFORM) # 使用全局变量
        # *******************************************
        
        gt_slice_int = np.round(gt_slice_oriented).astype(np.int16)

        for organ_name, channel_idx in organ_channel_map_local.items():
            if organ_name == 'kidney':
                gt_organ_binary_raw = ((gt_slice_int == 3) | (gt_slice_int == 4)).astype(np.float32)
            elif organ_name == 'liver':
                gt_organ_binary_raw = (gt_slice_int == 1).astype(np.float32)
            elif organ_name == 'spleen':
                gt_organ_binary_raw = (gt_slice_int == 2).astype(np.float32)
            elif organ_name == 'bowel':
                gt_organ_binary_raw = (gt_slice_int == 5).astype(np.float32)
            else:
                continue

            if gt_organ_binary_raw.shape != (target_size_local_int, target_size_local_int):
                gt_organ_resized = cv2.resize(gt_organ_binary_raw, 
                                              (target_size_local_int, target_size_local_int), 
                                              interpolation=cv2.INTER_NEAREST)
            else:
                gt_organ_resized = gt_organ_binary_raw
            
            gt_organ_resized = (gt_organ_resized > 0.5).astype(np.float32)
            pred_organ_binary_channel = pred_mask_binary[..., channel_idx]

            if np.sum(gt_organ_resized) > 0 or np.sum(pred_organ_binary_channel) > 0:
                dice = (2. * np.sum(gt_organ_resized * pred_organ_binary_channel) + 1e-6) / \
                       (np.sum(gt_organ_resized) + np.sum(pred_organ_binary_channel) + 1e-6)
                intersection = np.sum(gt_organ_resized * pred_organ_binary_channel)
                union = np.sum(gt_organ_resized) + np.sum(pred_organ_binary_channel) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                local_dice_scores[organ_name].append(dice)
                local_iou_scores[organ_name].append(iou)
        local_processed += 1
    return local_dice_scores, local_iou_scores, local_processed

def evaluate_segmentation_results(prediction_dir, segmentation_map, image_paths_dict,
                                 organ_map_nii, organ_channel_map, num_organs, target_size,
                                 prediction_threshold):
    """评估分割结果与真实掩码的匹配程度 (优化版)"""
    print("开始评估分割结果...")

    # 初始化分数记录
    dice_scores = {organ: [] for organ in organ_channel_map.keys()}
    iou_scores = {organ: [] for organ in organ_channel_map.keys()}

    # 获取有真实掩码的患者ID列表
    patient_ids_with_gt = list(segmentation_map.keys())
    print(f"找到 {len(patient_ids_with_gt)} 个有真实掩码的患者用于评估")

    processed_slices = 0
    
    # 使用ThreadPoolExecutor并行处理多个患者
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        
        for patient_id in patient_ids_with_gt:
            nii_path = segmentation_map.get(patient_id)
            dicom_paths = image_paths_dict.get(patient_id)
            if not nii_path or not dicom_paths: 
                continue

            # 查找该患者的预测文件
            pred_patient_dir = os.path.join(prediction_dir, patient_id)
            if not os.path.exists(pred_patient_dir):
                continue

            # 获取该病人第一个有预测文件的series
            series_dirs = [os.path.join(pred_patient_dir, d) for d in os.listdir(pred_patient_dir)
                           if os.path.isdir(os.path.join(pred_patient_dir, d))]
            if not series_dirs:
                continue
                
            series_pred_dir = series_dirs[0] # 假设评估第一个找到的series
            pred_files = glob.glob(os.path.join(series_pred_dir, "*.npz"))
            if not pred_files:
                continue
                
            # 提交任务到线程池
            futures.append(executor.submit(
                process_patient_evaluation, 
                patient_id, nii_path, dicom_paths, series_pred_dir, pred_files,
                organ_map_nii, organ_channel_map, target_size, prediction_threshold
            ))
        
        # 收集结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评估患者"):
            try:
                patient_dice_scores, patient_iou_scores, patient_processed = future.result()
                
                # 合并结果
                for organ in organ_channel_map.keys():
                    dice_scores[organ].extend(patient_dice_scores[organ])
                    iou_scores[organ].extend(patient_iou_scores[organ])
                    
                processed_slices += patient_processed
            except Exception as e:
                print(f"处理评估结果时出错: {e}")

    # --- 输出和绘制结果 ---
    print(f"\n评估完成，共处理 {processed_slices} 个有效切片。")
    print("=== 分割评估结果 (平均 Dice 和 IoU) ===")
    all_dices = []
    all_ious = []
    for organ in organ_channel_map.keys():
        mean_dice = np.mean(dice_scores[organ]) if dice_scores[organ] else 0
        mean_iou = np.mean(iou_scores[organ]) if iou_scores[organ] else 0
        print(f"  {organ}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, 样本数={len(dice_scores[organ])}")
        all_dices.extend(dice_scores[organ])
        all_ious.extend(iou_scores[organ])

    overall_mean_dice = np.mean(all_dices) if all_dices else 0
    overall_mean_iou = np.mean(all_ious) if all_ious else 0
    print(f"\n  总体平均: Dice={overall_mean_dice:.4f}, IoU={overall_mean_iou:.4f}, 总样本数={len(all_dices)}")

    # --- 绘制评估结果箱线图 ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    labels = list(organ_channel_map.keys())

    # Dice 分数箱线图
    dice_data_for_plot = [dice_scores[organ] for organ in labels]
    ax[0].boxplot(dice_data_for_plot, labels=labels, showfliers=False) # showfliers=False 隐藏异常值
    ax[0].set_title('各器官 Dice 系数分布')
    ax[0].set_ylabel('Dice 系数')
    ax[0].grid(True)

    # IoU 分数箱线图
    iou_data_for_plot = [iou_scores[organ] for organ in labels]
    ax[1].boxplot(iou_data_for_plot, labels=labels, showfliers=False)
    ax[1].set_title('各器官 IoU 分数分布')
    ax[1].set_ylabel('IoU 分数')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "segmentation_evaluation_boxplot.png"))
    plt.show()

    return dice_scores, iou_scores



def predict_and_save_masks_with_resume(model, patient_ids, image_paths_dict, output_dir, batch_size=32, 
                                      chunk_size=50, save_zips=True):
    """
    支持断点续传的掩码预测和保存函数，从已解压的目录读取已有结果
    
    参数:
    - chunk_size: 每个输出ZIP文件中包含的患者数量
    - save_zips: 是否将结果保存为ZIP文件
    """
    import shutil
    import time
    from datetime import datetime
    
    processed_slices_total = 0
    failed_saves = 0
    current_chunk_patients = 0
    current_chunk_num = 1
    processed_patients = []
    
    # 检查磁盘空间
    def check_disk_space(path='/kaggle/working'):
        import os
        stat = os.statvfs(path)
        free_bytes = stat.f_frsize * stat.f_bavail
        free_gb = free_bytes / (1024 ** 3)
        return free_gb
    
    # 已有预测结果目录
    previous_results_dir = '/kaggle/input/rsna-uneted-output/segmentation_predictions_multi_v2'
    has_previous_results = os.path.exists(previous_results_dir)
    
    # 检查是否有已完成患者记录
    completed_patients_file = '/kaggle/working/completed_patients.txt'
    completed_patients = set()
    
    # 从已有预测结果目录中识别已处理的患者
    if has_previous_results:
        print(f"发现已有预测结果目录: {previous_results_dir}")
        
        # 获取目录中的所有患者ID
        try:
            patient_dirs = [d for d in os.listdir(previous_results_dir) 
                          if os.path.isdir(os.path.join(previous_results_dir, d))]
            
            # 更新已完成患者集合
            for patient_id in patient_dirs:
                completed_patients.add(patient_id)
            
            print(f"从已有结果目录中识别出 {len(completed_patients)} 个已处理的患者")
            
            # 保存已完成患者列表
            with open(completed_patients_file, 'w') as f:
                for patient_id in sorted(completed_patients):
                    f.write(f"{patient_id}\n")
        except Exception as e:
            print(f"读取已有结果目录失败: {e}")
            has_previous_results = False
    
    # 创建时间戳，用于命名输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理每个患者
    for i, patient_id in enumerate(tqdm(patient_ids, desc="处理患者")):
        # 跳过已完成的患者
        if patient_id in completed_patients:
            print(f"跳过已完成的患者: {patient_id}")
            continue
            
        if patient_id not in image_paths_dict:
            continue
            
        dicom_paths = image_paths_dict[patient_id]
        if not dicom_paths:
            continue
        
        # 检查剩余磁盘空间
        free_space_gb = check_disk_space()
        if free_space_gb < 1.0:  # 如果剩余空间小于1GB
            print(f"警告: 磁盘空间不足 ({free_space_gb:.2f}GB)，准备保存当前进度并清理...")
            
            if save_zips and processed_patients:
                # 保存当前批次的结果
                chunk_zip_path = f'/kaggle/working/predictions_chunk_{current_chunk_num}_{timestamp}.zip'
                print(f"保存当前批次结果到: {chunk_zip_path}")
                
                # 创建ZIP文件
                import zipfile
                with zipfile.ZipFile(chunk_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for pid in processed_patients:
                        patient_dir = os.path.join(output_dir, pid)
                        if os.path.exists(patient_dir):
                            # 添加患者目录到ZIP
                            for root, _, files in os.walk(patient_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, output_dir)
                                    zipf.write(file_path, arc_name)
                
                # 更新已完成患者记录
                with open(completed_patients_file, 'a') as f:
                    for pid in processed_patients:
                        f.write(f"{pid}\n")
                        completed_patients.add(pid)
                
                # 清理已保存的患者目录
                for pid in processed_patients:
                    patient_dir = os.path.join(output_dir, pid)
                    if os.path.exists(patient_dir):
                        shutil.rmtree(patient_dir)
                
                # 重置处理状态
                processed_patients = []
                current_chunk_patients = 0
                current_chunk_num += 1
                
                print(f"已保存并清理空间，继续处理...")
            else:
                print("警告: 磁盘空间不足，但没有启用ZIP保存或没有已处理患者，继续尝试...")
            
        # 获取该患者的series_id (假设所有图像来自同一个系列)
        series_id = dicom_paths[0].split('/')[-2]
        
        # 创建输出目录
        patient_output_dir = os.path.join(output_dir, patient_id, series_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # 如果有已有结果，从已有目录中复制该患者的掩码
        if has_previous_results:
            prev_patient_dir = os.path.join(previous_results_dir, patient_id)
            if os.path.exists(prev_patient_dir):
                prev_series_dirs = [d for d in os.listdir(prev_patient_dir) 
                                  if os.path.isdir(os.path.join(prev_patient_dir, d))]
                
                if prev_series_dirs:
                    prev_series_dir = os.path.join(prev_patient_dir, prev_series_dirs[0])  # 使用第一个系列
                    mask_files = glob.glob(os.path.join(prev_series_dir, "*.npz"))
                    
                    if mask_files:
                        print(f"在已有结果目录中找到患者 {patient_id} 的 {len(mask_files)} 个掩码文件")
                        
                        # 如果掩码文件数量与DICOM文件数量相同，则认为该患者已完全处理
                        if len(mask_files) >= len(dicom_paths):
                            print(f"患者 {patient_id} 在已有结果中已完全处理，标记为完成并跳过")
                            completed_patients.add(patient_id)
                            with open(completed_patients_file, 'a') as f:
                                f.write(f"{patient_id}\n")
                            continue
                        
                        # 复制已有的掩码文件
                        for mask_file in mask_files:
                            mask_filename = os.path.basename(mask_file)
                            dest_path = os.path.join(patient_output_dir, mask_filename)
                            
                            if not os.path.exists(dest_path):
                                try:
                                    shutil.copy2(mask_file, dest_path)
                                    processed_slices_total += 1
                                except Exception as e:
                                    print(f"复制掩码文件失败 {mask_file} -> {dest_path}: {e}")
        
        # 收集需要处理的切片
        slices_to_process = []
        instance_numbers = []
        
        # 检查当前输出目录中已有的掩码
        existing_masks = glob.glob(os.path.join(patient_output_dir, "*.npz"))
        existing_instances = {os.path.splitext(os.path.basename(m))[0] for m in existing_masks}
        
        for dicom_path in dicom_paths:
            # 加载DICOM并获取实例号
            image, instance_num = load_dicom_slice(dicom_path)
            if image is None:
                continue
                
            # 使用实例号作为文件名
            instance_number = instance_num if instance_num is not None else dicom_paths.index(dicom_path) + 1
            instance_number_str = str(instance_number)
            
            # 检查该切片是否已处理
            if instance_number_str in existing_instances:
                continue
            
            # 预处理图像
            processed_image = preprocess_image_for_unet(image, TARGET_SIZE)
            slices_to_process.append(processed_image)
            instance_numbers.append(instance_number)
            
        if not slices_to_process:
            print(f"患者 {patient_id} 的所有切片已处理完成")
            # 标记该患者为已完成
            completed_patients.add(patient_id)
            with open(completed_patients_file, 'a') as f:
                f.write(f"{patient_id}\n")
            continue
            
        print(f"处理患者 {patient_id}: {len(slices_to_process)}/{len(dicom_paths)} 个新切片")
        
        # 批量预测
        try:
            for i in range(0, len(slices_to_process), batch_size):
                batch_images = np.array(slices_to_process[i:i+batch_size])
                batch_predictions = model.predict(batch_images, verbose=0)
                
                # 保存预测结果
                for j, pred in enumerate(batch_predictions):
                    idx = i + j
                    if idx >= len(instance_numbers):
                        break
                        
                    instance_number = instance_numbers[idx]
                    output_path = os.path.join(patient_output_dir, f"{instance_number}.npz")
                    
                    # 转换为二值掩码
                    pred_mask_binary = (pred > PREDICTION_THRESHOLD).astype(np.uint8)
                    
                    try:
                        np.savez_compressed(output_path, mask=pred_mask_binary)
                        processed_slices_total += 1
                    except Exception as e:
                        print(f"保存掩码失败 {output_path}: {e}")
                        failed_saves += 1
                        
                # 释放内存
                del batch_images
                del batch_predictions
                gc.collect()
                
            # 该患者处理完毕，添加到处理列表
            processed_patients.append(patient_id)
            current_chunk_patients += 1
            
            # 如果达到块大小，保存并清理
            if save_zips and current_chunk_patients >= chunk_size:
                import zipfile
                chunk_zip_path = f'/kaggle/working/predictions_chunk_{current_chunk_num}_{timestamp}.zip'
                print(f"保存当前批次结果到: {chunk_zip_path}")
                
                # 创建ZIP文件
                with zipfile.ZipFile(chunk_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for pid in processed_patients:
                        patient_dir = os.path.join(output_dir, pid)
                        if os.path.exists(patient_dir):
                            # 添加患者目录到ZIP
                            for root, _, files in os.walk(patient_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arc_name = os.path.relpath(file_path, output_dir)
                                    zipf.write(file_path, arc_name)
                
                # 更新已完成患者记录
                with open(completed_patients_file, 'a') as f:
                    for pid in processed_patients:
                        f.write(f"{pid}\n")
                        completed_patients.add(pid)
                
                # 清理已保存的患者目录
                for pid in processed_patients:
                    patient_dir = os.path.join(output_dir, pid)
                    if os.path.exists(patient_dir):
                        shutil.rmtree(patient_dir)
                
                # 重置处理状态
                processed_patients = []
                current_chunk_patients = 0
                current_chunk_num += 1
                
        except Exception as e:
            print(f"批量预测失败: {e}")
            # 尝试保存当前进度
            if processed_patients:
                try:
                    import zipfile
                    chunk_zip_path = f'/kaggle/working/predictions_partial_{current_chunk_num}_{timestamp}.zip'
                    print(f"保存当前进度到: {chunk_zip_path}")
                    
                    with zipfile.ZipFile(chunk_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for pid in processed_patients:
                            patient_dir = os.path.join(output_dir, pid)
                            if os.path.exists(patient_dir):
                                for root, _, files in os.walk(patient_dir):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arc_name = os.path.relpath(file_path, output_dir)
                                        zipf.write(file_path, arc_name)
                    
                    # 更新已完成患者记录
                    with open(completed_patients_file, 'a') as f:
                        for pid in processed_patients:
                            if pid != patient_id:  # 不包括当前失败的患者
                                f.write(f"{pid}\n")
                                completed_patients.add(pid)
                except Exception as save_err:
                    print(f"保存进度失败: {save_err}")
    
    # 保存最后一批结果
    if save_zips and processed_patients:
        import zipfile
        final_zip_path = f'/kaggle/working/predictions_final_{timestamp}.zip'
        print(f"保存最终批次结果到: {final_zip_path}")
        
        with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pid in processed_patients:
                patient_dir = os.path.join(output_dir, pid)
                if os.path.exists(patient_dir):
                    for root, _, files in os.walk(patient_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arc_name)
        
        # 更新已完成患者记录
        with open(completed_patients_file, 'a') as f:
            for pid in processed_patients:
                f.write(f"{pid}\n")
    
    return processed_slices_total, failed_saves


def prepare_for_evaluation(prediction_dir, previous_results_dir=None, zip_files=None):
    """
    准备评估环境，合并所有预测结果
    
    参数:
    - prediction_dir: 当前预测结果目录
    - previous_results_dir: 已有预测结果目录
    - zip_files: 包含预测结果的ZIP文件列表
    
    返回:
    - merged_dir: 合并后的预测结果目录
    """
    import shutil
    import tempfile
    import zipfile
    
    print("准备评估环境，合并所有预测结果...")
    
    # 创建合并目录
    merged_dir = os.path.join('/kaggle/working', 'merged_predictions')
    os.makedirs(merged_dir, exist_ok=True)
    
    # 1. 首先复制当前预测目录中的结果
    if os.path.exists(prediction_dir):
        print(f"复制当前预测结果从 {prediction_dir}")
        for patient_id in os.listdir(prediction_dir):
            patient_src = os.path.join(prediction_dir, patient_id)
            patient_dst = os.path.join(merged_dir, patient_id)
            
            if os.path.isdir(patient_src) and not os.path.exists(patient_dst):
                shutil.copytree(patient_src, patient_dst)
    
    # 2. 复制已有预测结果目录中的内容
    if previous_results_dir and os.path.exists(previous_results_dir):
        print(f"复制已有预测结果从 {previous_results_dir}")
        for patient_id in os.listdir(previous_results_dir):
            patient_src = os.path.join(previous_results_dir, patient_id)
            patient_dst = os.path.join(merged_dir, patient_id)
            
            # 如果目标中已有该患者，则跳过
            if os.path.isdir(patient_src) and not os.path.exists(patient_dst):
                shutil.copytree(patient_src, patient_dst)
    
    # 3. 处理ZIP文件中的内容
    if zip_files:
        for zip_path in zip_files:
            if not os.path.exists(zip_path):
                continue
                
            print(f"从ZIP文件中提取预测结果: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    # 获取ZIP中的患者目录
                    all_files = zipf.namelist()
                    patient_dirs = set()
                    
                    for file_path in all_files:
                        parts = file_path.split('/')
                        if len(parts) >= 1:
                            patient_dirs.add(parts[0])
                    
                    # 提取每个患者的文件
                    for patient_id in patient_dirs:
                        # 如果合并目录中已有该患者，则跳过
                        patient_dst = os.path.join(merged_dir, patient_id)
                        if os.path.exists(patient_dst):
                            continue
                            
                        # 创建临时目录提取文件
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # 提取该患者的所有文件
                            for file_info in zipf.infolist():
                                if file_info.filename.startswith(f"{patient_id}/"):
                                    zipf.extract(file_info, temp_dir)
                            
                            # 复制到合并目录
                            patient_temp = os.path.join(temp_dir, patient_id)
                            if os.path.exists(patient_temp):
                                shutil.copytree(patient_temp, patient_dst)
            except Exception as e:
                print(f"处理ZIP文件 {zip_path} 失败: {e}")
    
    # 统计合并后的患者数量
    patient_count = len([d for d in os.listdir(merged_dir) if os.path.isdir(os.path.join(merged_dir, d))])
    print(f"合并完成，共有 {patient_count} 个患者的预测结果")
    
    return merged_dir




def predict_and_save_masks_optimized(model, patient_ids, image_paths_dict, output_dir, batch_size=32):
    """
    优化版的掩码预测和保存函数，不包含恢复逻辑，专注于处理新患者
    
    参数:
    - model: 训练好的模型
    - patient_ids: 需要处理的患者ID列表
    - image_paths_dict: 患者ID到DICOM路径列表的映射
    - output_dir: 输出目录
    - batch_size: 批处理大小
    
    返回:
    - processed_slices_total: 处理的切片总数
    - failed_saves: 保存失败的次数
    """
    processed_slices_total = 0
    failed_saves = 0
    
    for patient_id in tqdm(patient_ids, desc="处理患者"):
        if patient_id not in image_paths_dict:
            continue
            
        dicom_paths = image_paths_dict[patient_id]
        if not dicom_paths:
            continue
            
        # 获取该患者的series_id (假设所有图像来自同一个系列)
        series_id = dicom_paths[0].split('/')[-2]
        
        # 创建输出目录
        patient_output_dir = os.path.join(output_dir, patient_id, series_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # 收集需要处理的切片
        slices_to_process = []
        instance_numbers = []
        
        for dicom_path in dicom_paths:
            # 加载DICOM并获取实例号
            image, instance_num = load_dicom_slice(dicom_path)
            if image is None:
                continue
                
            # 使用实例号作为文件名
            instance_number = instance_num if instance_num is not None else dicom_paths.index(dicom_path) + 1
            
            # 预处理图像
            processed_image = preprocess_image_for_unet(image, TARGET_SIZE)
            slices_to_process.append(processed_image)
            instance_numbers.append(instance_number)
            
        if not slices_to_process:
            continue
            
        print(f"处理患者 {patient_id}: {len(slices_to_process)} 个切片")
        
        # 批量预测
        try:
            for i in range(0, len(slices_to_process), batch_size):
                batch_images = np.array(slices_to_process[i:i+batch_size])
                batch_predictions = model.predict(batch_images, verbose=0)
                
                # 保存预测结果
                for j, pred in enumerate(batch_predictions):
                    idx = i + j
                    if idx >= len(instance_numbers):
                        break
                        
                    instance_number = instance_numbers[idx]
                    output_path = os.path.join(patient_output_dir, f"{instance_number}.npz")
                    
                    # 转换为二值掩码
                    pred_mask_binary = (pred > PREDICTION_THRESHOLD).astype(np.uint8)
                    
                    try:
                        np.savez_compressed(output_path, mask=pred_mask_binary)
                        processed_slices_total += 1
                    except Exception as e:
                        print(f"保存掩码失败 {output_path}: {e}")
                        failed_saves += 1
                        
                # 释放内存
                del batch_images
                del batch_predictions
                gc.collect()
                
        except Exception as e:
            print(f"批量预测失败: {e}")
    
    return processed_slices_total, failed_saves



def modified_evaluation_approach(best_model, image_paths_dict, segmentation_map):
    """不合并结果，直接评估已有预测和新生成的预测"""
    
    # 1. 识别已有预测结果中的患者
    previous_results_dir = '/kaggle/input/rsna-uneted-output/merged_predictions'
    if os.path.exists(previous_results_dir):
        existing_patients = set(os.listdir(previous_results_dir))
        print(f"已有预测结果中包含 {len(existing_patients)} 个患者")
    else:
        existing_patients = set()
        print("未找到已有预测结果")
    
    # 2. 确定需要处理的患者
    all_patient_ids = sorted([pid for pid in os.listdir(TRAIN_IMAGES_DIR) 
                             if os.path.isdir(os.path.join(TRAIN_IMAGES_DIR, pid))])
    
    patients_to_process = [pid for pid in all_patient_ids if pid not in existing_patients]
    print(f"需要处理的患者数量: {len(patients_to_process)}")
    
    # 3. 处理剩余患者
    if patients_to_process:
        print("\n--- 处理剩余患者 ---")
        processed_slices, failed_saves = predict_and_save_masks_optimized(
            best_model, patients_to_process, image_paths_dict, 
            PREDICTION_OUTPUT_DIR, INFERENCE_BATCH_SIZE
        )
        print(f"新处理的切片数量: {processed_slices}")
        print(f"保存失败的切片数量: {failed_saves}")
    
    # 4. 直接评估结果 (不复制或合并文件)
    print("\n--- 评估分割结果 ---")
    
    # 初始化分数记录
    dice_scores = {organ: [] for organ in ORGAN_CHANNEL_MAP.keys()}
    iou_scores = {organ: [] for organ in ORGAN_CHANNEL_MAP.keys()}
    
    # 获取有真实掩码的患者ID列表
    patient_ids_with_gt = list(segmentation_map.keys())
    print(f"找到 {len(patient_ids_with_gt)} 个有真实掩码的患者用于评估")
    processed_patients = 0
    
    # 使用ThreadPoolExecutor并行处理多个患者
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        
        for patient_id in patient_ids_with_gt:
            nii_path = segmentation_map.get(patient_id)
            dicom_paths = image_paths_dict.get(patient_id)
            if not nii_path or not dicom_paths: 
                continue
            
            # 确定该患者的预测掩码目录
            if patient_id in existing_patients:
                pred_patient_dir = os.path.join(previous_results_dir, patient_id)
            else:
                pred_patient_dir = os.path.join(PREDICTION_OUTPUT_DIR, patient_id)
                
            if not os.path.exists(pred_patient_dir):
                continue
                
            # 获取该患者的系列目录
            series_dirs = [d for d in os.listdir(pred_patient_dir) 
                          if os.path.isdir(os.path.join(pred_patient_dir, d))]
            if not series_dirs:
                continue
                
            # 使用第一个系列
            series_pred_dir = os.path.join(pred_patient_dir, series_dirs[0])
            pred_files = glob.glob(os.path.join(series_pred_dir, "*.npz"))
            if not pred_files:
                continue
                
            # 提交任务到线程池
            futures.append(executor.submit(
                process_patient_evaluation, 
                patient_id, nii_path, dicom_paths, series_pred_dir, pred_files,
                ORGAN_MAP_NII, ORGAN_CHANNEL_MAP, TARGET_SIZE, PREDICTION_THRESHOLD
            ))
        
        # 收集结果
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="评估患者"):
            try:
                patient_dice_scores, patient_iou_scores, patient_processed = future.result()
                
                # 合并结果
                for organ in ORGAN_CHANNEL_MAP.keys():
                    dice_scores[organ].extend(patient_dice_scores[organ])
                    iou_scores[organ].extend(patient_iou_scores[organ])
                    
                processed_patients += 1
            except Exception as e:
                print(f"处理评估结果时出错: {e}")
    
    # 输出评估结果
    print(f"\n评估完成，成功评估 {processed_patients} 个患者。")
    print("=== 分割评估结果 (平均 Dice 和 IoU) ===")
    all_dices = []
    all_ious = []
    for organ in ORGAN_CHANNEL_MAP.keys():
        mean_dice = np.mean(dice_scores[organ]) if dice_scores[organ] else 0
        mean_iou = np.mean(iou_scores[organ]) if iou_scores[organ] else 0
        print(f"  {organ}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, 样本数={len(dice_scores[organ])}")
        all_dices.extend(dice_scores[organ])
        all_ious.extend(iou_scores[organ])

    overall_mean_dice = np.mean(all_dices) if all_dices else 0
    overall_mean_iou = np.mean(all_ious) if all_ious else 0
    print(f"\n  总体平均: Dice={overall_mean_dice:.4f}, IoU={overall_mean_iou:.4f}, 总样本数={len(all_dices)}")
    
    # 绘制结果
    try:
        # --- 绘制评估结果箱线图 ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        labels = list(ORGAN_CHANNEL_MAP.keys())

        # Dice 分数箱线图
        dice_data_for_plot = [dice_scores[organ] for organ in labels]
        ax[0].boxplot(dice_data_for_plot, labels=labels, showfliers=False) # showfliers=False 隐藏异常值
        ax[0].set_title('各器官 Dice 系数分布')
        ax[0].set_ylabel('Dice 系数')
        ax[0].grid(True)

        # IoU 分数箱线图
        iou_data_for_plot = [iou_scores[organ] for organ in labels]
        ax[1].boxplot(iou_data_for_plot, labels=labels, showfliers=False)
        ax[1].set_title('各器官 IoU 分数分布')
        ax[1].set_ylabel('IoU 分数')
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "segmentation_evaluation_boxplot.png"))
        plt.show()
    except Exception as e:
        print(f"绘图失败: {e}")
    
    return dice_scores, iou_scores



def main():
    """主程序流程 (优化版)"""
    print("--- 1. 构建文件映射 ---")
    # 加载系列元数据
    train_meta_path = os.path.join(DATA_DIR, 'train_series_meta.csv')
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"找不到系列元数据文件: {train_meta_path}")
    train_series_meta = pd.read_csv(train_meta_path)

    # 创建 series_id -> patient_id 映射
    series_to_patient = dict(zip(
        train_series_meta['series_id'].astype(str),
        train_series_meta['patient_id'].astype(str)
    ))

    # 创建 NII 分割文件映射 (series_id -> nii_path)
    series_to_nii = {}
    segmentation_files = glob.glob(os.path.join(SEGMENTATION_DIR, "*.nii"))
    print(f"发现 {len(segmentation_files)} 个 NII 文件。")
    for fpath in segmentation_files:
        series_id = os.path.splitext(os.path.basename(fpath))[0]
        if series_id in series_to_patient: # 确保这个系列在元数据中
            series_to_nii[series_id] = fpath

    print(f"成功映射 {len(series_to_nii)} 个 NII 文件到 series_id。")

    # 加载 DICOM tags (如果可用)
    dicom_tags_df = None
    dicom_tags_path = os.path.join(DATA_DIR, 'train_dicom_tags.parquet')
    if os.path.exists(dicom_tags_path):
        print("加载 DICOM tags...")
        try:
            dicom_tags_df = pd.read_parquet(dicom_tags_path)
            # 预处理 tags DataFrame
            if 'PatientID' in dicom_tags_df.columns:
                dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
            # 尝试提取 series_id
            if 'SeriesInstanceUID' in dicom_tags_df.columns:
                 dicom_tags_df['series_id_extracted'] = dicom_tags_df['SeriesInstanceUID'].str.split('.').str[-2]
                 dicom_tags_df = dicom_tags_df.dropna(subset=['series_id_extracted'])
            print("DICOM tags 加载完成。")
        except Exception as e:
            print(f"加载 DICOM tags 失败: {e}. 将不使用 tags 进行排序。")
            dicom_tags_df = None
    else:
        print("未找到 DICOM tags 文件，将仅依赖DICOM头或文件名排序。")

    # --- 关联图像和分割 ---
    image_paths_dict = {}  # {patient_id: [sorted_list_of_dicom_paths]}
    segmentation_map = {}  # {patient_id: nii_path}
    
    patient_ids_in_train_images = os.listdir(TRAIN_IMAGES_DIR)
    valid_patients = []  # 存储有图像和对应NII文件的患者ID

    print("开始关联患者图像和分割文件...")
    for patient_id in tqdm(patient_ids_in_train_images, desc="处理患者"):
        if not os.path.isdir(os.path.join(TRAIN_IMAGES_DIR, patient_id)):
            continue

        patient_series_ids = [s for s, p in series_to_patient.items() if p == patient_id]
        if not patient_series_ids: 
            continue

        # 查找该患者是否有系列同时存在于图像目录和NII分割中
        found_valid_series = False
        for series_id in patient_series_ids:
            series_img_path = os.path.join(TRAIN_IMAGES_DIR, patient_id, series_id)
            nii_path = series_to_nii.get(series_id)

            if os.path.isdir(series_img_path) and nii_path:
                # 获取并排序DICOM文件
                dicom_info = get_dicom_files_dict(patient_id, series_id, dicom_tags_df)
                if dicom_info:  # 确保系列中有有效的DICOM文件
                    image_paths_dict[patient_id] = [item[1] for item in dicom_info]  # 只存路径
                    segmentation_map[patient_id] = nii_path
                    valid_patients.append(patient_id)
                    found_valid_series = True
                    break  # 每个患者只使用一个有效的 series 和 NII

    final_patient_ids = sorted(list(set(valid_patients)))  # 去重并排序
    print(f"成功映射了 {len(final_patient_ids)} 位患者的图像和分割文件。")
    if not final_patient_ids:
        raise SystemExit("错误：未能找到任何包含有效图像序列及对应NII文件的患者。")
        
    print("\n--- 2. 验证NII文件标签 ---")
    # 使用修正后的器官映射进行验证
    verify_nii_labels(segmentation_map, {1: 'liver', 2: 'spleen', 3: 'kidney_left', 4: 'kidney_right', 5: 'bowel'})

    print("\n--- 3. 分析器官分布 ---")
    # 使用模型将要使用的器官映射进行分析 (合并肾脏)
    distribution_results = analyze_organ_distribution(segmentation_map, ORGAN_MAP_NII)
    calculated_class_weights = distribution_results['class_weights']  # 保存计算出的权重

    print("\n--- 4. 划分训练集和验证集 ---")
    train_pids, val_pids = train_test_split(final_patient_ids, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
    print(f"训练集患者数: {len(train_pids)}")
    print(f"验证集患者数: {len(val_pids)}")

    print("\n--- 5. 使用已有预处理数据 ---")
    # 检查预处理数据目录
    preprocessed_patients = [d for d in os.listdir(PREPROCESSED_DIR) 
                          if os.path.isdir(os.path.join(PREPROCESSED_DIR, d))]
    print(f"找到 {len(preprocessed_patients)} 个已预处理的患者数据")
    
    # 添加这一行，在训练前检查预处理数据的对齐情况
    print("\n--- 5b. 检查预处理数据对齐 ---")
    visualize_preprocessed_samples(PREPROCESSED_DIR, num_patients=3, samples_per_patient=2)
        
    # 确认预处理的患者包含了训练和验证集
    train_pids_preprocessed = [pid for pid in train_pids if pid in preprocessed_patients]
    val_pids_preprocessed = [pid for pid in val_pids if pid in preprocessed_patients]
    
    print(f"预处理后的训练集患者数: {len(train_pids_preprocessed)}/{len(train_pids)}")
    print(f"预处理后的验证集患者数: {len(val_pids_preprocessed)}/{len(val_pids)}")
        
    if len(train_pids_preprocessed) == 0 or len(val_pids_preprocessed) == 0:
        raise SystemExit("错误：预处理后训练集或验证集为空。")
    
    # --- 准备训练参数 ---
    epochs_config = [EPOCHS_STAGE1, EPOCHS_STAGE2, EPOCHS_STAGE3, EPOCHS_STAGE4]
    
    print("\n--- 6. 开始分阶段训练模型 ---")
    stage_histories = train_in_stages(
        train_pids_preprocessed, val_pids_preprocessed, PREPROCESSED_DIR,
        TARGET_SIZE, N_INPUT_CHANNELS, NUM_ORGANS,
        BATCH_SIZE, LEARNING_RATE, epochs_config, calculated_class_weights
    )
    
    # --- 7. 绘制训练历史曲线 ---
    print("\n--- 7. 绘制训练历史曲线 ---")
    plt.figure(figsize=(18, 12))
    num_stages = len(stage_histories)
    colors = plt.cm.viridis(np.linspace(0, 1, num_stages))
    
    # 绘制各阶段的平均Dice系数
    plt.subplot(2, 2, 1)
    for i, (stage_name, history) in enumerate(stage_histories.items()):
        epochs = range(1, len(history['average_dice_coefficient']) + 1)
        plt.plot(epochs, history['average_dice_coefficient'], label=f'{stage_name} Train Avg Dice', color=colors[i], linestyle='--')
        if 'val_average_dice_coefficient' in history:
             plt.plot(epochs, history['val_average_dice_coefficient'], label=f'{stage_name} Val Avg Dice', color=colors[i])
    plt.title('平均Dice系数 (所有阶段)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice系数')
    plt.legend()
    plt.grid(True)
    
    # 绘制各阶段的损失
    plt.subplot(2, 2, 2)
    for i, (stage_name, history) in enumerate(stage_histories.items()):
         epochs = range(1, len(history['loss']) + 1)
         plt.plot(epochs, history['loss'], label=f'{stage_name} Train Loss', color=colors[i], linestyle='--')
         if 'val_loss' in history:
              plt.plot(epochs, history['val_loss'], label=f'{stage_name} Val Loss', color=colors[i])
    plt.title('损失 (所有阶段)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制最终阶段的各器官验证Dice系数
    plt.subplot(2, 2, 3)
    final_stage_name = list(stage_histories.keys())[-1]
    final_history = stage_histories[final_stage_name]
    epochs = range(1, len(final_history['val_dice_liver']) + 1) # 假设所有指标长度相同
    plotplt(epochs, final_history['val_dice_liver'], label='肝脏 (Val)')
    plt.plot(epochs, final_history['val_dice_spleen'], label='脾脏 (Val)')
    plt.plot(epochs, final_historyval['_dice_kidney'], label='肾脏 (Val)')
    plt.plot(epochs, final_history['val_dice_bowel'], label='肠道 (Val)')
    plt.title(f'各器官 Dice 系数 ({final_stage_name} - 验证集)')
    plt.xlabel('Epochs')
    plt.ylabel('Dice系数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history_all_stages_v2.png"))
    plt.show()

    # --- 8. 加载最终最佳模型进行推理 ---
    print("\n--- 8. 加载最终最佳模型进行推理 ---")
    ordered_class_weights_for_load = [calculated_class_weights.get(organ, 1.0) for organ in ORGAN_CHANNEL_MAP.keys()]
    final_loss_for_loading = create_focal_dice_loss( # 使用与训练时相同的参数
        gamma_focal=2.0, alpha_focal=0.25, 
        lambda_focal=0.5, lambda_dice=0.5, 
        class_weights=ordered_class_weights_for_load
    )
    
    custom_objects = {
        'focal_dice_loss_fn': final_loss_for_loading, # 使用创建函数返回的实际损失函数
        # 或者，如果损失函数被命名，使用那个名字，并在全局定义它
        'average_dice_coefficient': average_dice_coefficient,
        'dice_liver': dice_liver, 'dice_spleen': dice_spleen,
        'dice_kidney': dice_kidney, 'dice_bowel': dice_bowel
    }
    
    try:
        # TensorFlow有时可以直接反序列化函数对象，但更可靠的是传递名称或Loss类
        # 如果上面的 focal_dice_loss_fn 不能直接被识别，
        # 你可能需要将 create_focal_dice_loss 返回的函数在全局命名，
        # 或者将 FocalDiceLoss 实现为一个 tf.keras.losses.Loss 的子类。
        # 鉴于之前的错误，我们先尝试加载时不编译或只带指标编译。
        best_model = models.load_model(MODEL_SAVE_PATH, custom_objects=custom_objects, compile=False) # 尝试 compile=False
        # 如果需要评估，后续再用优化器和损失函数编译一次
        best_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE*0.01), # 用一个小的学习率重新编译
                           loss=final_loss_for_loading, 
                           metrics=METRICS)
        print(f"成功加载并重新编译最终模型: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"加载最终模型 {MODEL_SAVE_PATH} 失败: {e}")
        print("确保 custom_objects 中的损失函数名称与保存时一致，或尝试仅加载权重。")
        # 如果完全失败，后续的推理和评估将无法进行
        return 


    # --- 9. 使用修改后的方法进行预测和评估 ---
    print("\n--- 9. 使用修改后的方法进行预测和评估 ---")

     # 获取已存在的预测结果患者列表
    previous_results_dir = '/kaggle/input/rsna-uneted-output/merged_predictions'
    if os.path.exists(previous_results_dir):
        existing_patients = set(os.listdir(previous_results_dir))
        print(f"已有预测结果中包含 {len(existing_patients)} 个患者")
    else:
        existing_patients = set()
        print("未找到已有预测结果")
    
    # 直接调用新的评估函数，它会处理未处理的患者并评估所有结果
    dice_scores, iou_scores = modified_evaluation_approach(best_model, image_paths_dict, segmentation_map)
    
    
    # --- 10. 可视化一些分割结果 ---
    print("\n--- 10. 可视化一些分割结果 ---")
    
    # 定义可视化时使用的颜色 (BGR格式，因为OpenCV常用，但显示时会转RGB)
    # 您可以根据 ORGAN_CHANNEL_MAP 的顺序调整或确保颜色字典覆盖所有需要的器官
    colors_for_visualization = { 
        'liver': [0, 0, 255],  # 红色 for liver
        'spleen': [0, 255, 0], # 绿色 for spleen
        'kidney': [255, 0, 0], # 蓝色 for kidney
        'bowel': [0, 255, 255]   # 黄色 for bowel (Cyan in BGR for Yellow in RGB)
    }

    # 从验证集或所有有效患者中随机选择几个进行可视化
    if val_pids and len(val_pids) > 0: # 优先使用验证集
        vis_patient_ids_options = val_pids
    elif 'final_patient_ids' in locals() and len(final_patient_ids) > 0:
        vis_patient_ids_options = final_patient_ids
    else:
        vis_patient_ids_options = list(image_paths_dict.keys()) # 如果都没有，则从所有有路径的患者中选

    if not vis_patient_ids_options:
        print("没有可供可视化的患者ID。")
    else:
        num_patients_to_visualize = min(3, len(vis_patient_ids_options)) # 最多可视化3个患者
        vis_patient_ids = np.random.choice(vis_patient_ids_options, num_patients_to_visualize, replace=False)
        print(f"将可视化以下患者的随机切片: {vis_patient_ids}")

        for patient_id_vis in vis_patient_ids: # 使用新的变量名避免冲突
            if patient_id_vis not in image_paths_dict or patient_id_vis not in segmentation_map:
                print(f"跳过患者 {patient_id_vis}: 缺少图像路径或分割图信息。")
                continue

            dicom_paths_current_patient = image_paths_dict[patient_id_vis] # 这是 (instance_num, path) 的列表
            nii_path_current_patient = segmentation_map[patient_id_vis]

            if not dicom_paths_current_patient:
                print(f"患者 {patient_id_vis} 的DICOM路径列表为空。")
                continue
            
            # 随机选择一个DICOM切片进行可视化
            # dicom_list_idx 是选中的DICOM在其排序列表中的索引
            dicom_list_idx_vis = np.random.randint(0, len(dicom_paths_current_patient))
            # 从 dicom_info_list 获取 instance_number 和 path
            # 假设 get_dicom_files_dict 返回的是 [(instance_number, path), ...]
            # 而 image_paths_dict[patient_id] 存储的是 [path, ...]
            # 需要统一：假设 image_paths_dict[patient_id] 就是 get_dicom_files_dict 返回的完整元组列表
            # 如果不是，需要调整 image_paths_dict 的构建方式，或者在这里重新获取instance_number
            
            # 假设 image_paths_dict[patient_id] 是路径列表，我们需要重新获取 InstanceNumber
            # 或者，如果您的 image_paths_dict 存储的是 (instance, path) 元组，则可以直接用
            # 为了代码的鲁棒性，我们重新读取一下InstanceNumber
            
            selected_dicom_path_vis = ""
            # 检查 image_paths_dict[patient_id_vis] 中元素的类型
            if isinstance(dicom_paths_current_patient[dicom_list_idx_vis], tuple): # 如果是 (inst, path)
                inst_num_vis, selected_dicom_path_vis = dicom_paths_current_patient[dicom_list_idx_vis]
            else: # 如果只是路径列表
                selected_dicom_path_vis = dicom_paths_current_patient[dicom_list_idx_vis]
                # 尝试从文件名或DICOM头读取InstanceNumber
                try:
                    temp_ds = pydicom.dcmread(selected_dicom_path_vis, stop_before_pixels=True)
                    inst_num_vis = int(temp_ds.InstanceNumber)
                except:
                    inst_num_vis = dicom_list_idx_vis + 1 # 备用方案
                    print(f"警告: 无法从 {selected_dicom_path_vis} 读取InstanceNumber，使用索引 {dicom_list_idx_vis} 作为替代ID。")


            print(f"\n可视化患者 {patient_id_vis}, DICOM列表索引 {dicom_list_idx_vis}, InstanceNumber {inst_num_vis}")

            image_vis_raw, _ = load_dicom_slice(selected_dicom_path_vis) # load_dicom_slice返回归一化图像和instance_number
            if image_vis_raw is None: 
                print(f"  无法加载DICOM: {selected_dicom_path_vis}")
                continue

            # --- 加载并处理真实NIFTI掩码 ---
            try:
                nii_img_vis = nib.load(nii_path_current_patient)
                gt_data_vis_float = nii_img_vis.get_fdata(dtype=np.float32)
                nii_total_slices_vis = gt_data_vis_float.shape[2]

                nii_slice_idx_for_gt = -1
                if USE_REVERSE_NIFTI_MAPPING: # 使用全局配置
                    nii_slice_idx_for_gt = nii_total_slices_vis - 1 - dicom_list_idx_vis
                else:
                    nii_slice_idx_for_gt = dicom_list_idx_vis
                
                if not (0 <= nii_slice_idx_for_gt < nii_total_slices_vis):
                    print(f"  错误: 为DICOM索引 {dicom_list_idx_vis} 计算的NIFTI索引 {nii_slice_idx_for_gt} 超出范围 ({nii_total_slices_vis}片)。跳过此切片。")
                    continue

                gt_slice_raw_from_nii = gt_data_vis_float[:, :, nii_slice_idx_for_gt]
                
                # 应用方向变换 (在原始分辨率下进行)
                gt_slice_oriented_raw_res = apply_orientation_transform(gt_slice_raw_from_nii, BEST_NIFTI_ORIENTATION_TRANSFORM)
                gt_slice_int_oriented_raw_res = np.round(gt_slice_oriented_raw_res).astype(np.int16)
                
            except Exception as e:
                print(f"  加载或处理真实NIFTI掩码时出错 (患者 {patient_id_vis}, NII: {nii_path_current_patient}): {e}")
                continue
            
            # --- 获取预测掩码 ---
            # 假设您的 image_paths_dict[patient_id] 中的路径是 /kaggle/input/.../patient_id/series_id/instance.dcm 格式
            series_id_vis = os.path.basename(os.path.dirname(selected_dicom_path_vis))
            
            pred_mask_npz_path = os.path.join(PREDICTION_OUTPUT_DIR, str(patient_id_vis), str(series_id_vis), f"{inst_num_vis}.npz")
            pred_mask_binary_vis = None # 初始化

            if os.path.exists(pred_mask_npz_path):
                try:
                    pred_data_vis = np.load(pred_mask_npz_path)
                    pred_mask_from_npz = pred_data_vis['mask'] # 假设保存的是概率或二值掩码
                    pred_mask_binary_vis = (pred_mask_from_npz > PREDICTION_THRESHOLD).astype(np.uint8)
                    if pred_mask_binary_vis.shape != (TARGET_SIZE, TARGET_SIZE, NUM_ORGANS):
                         print(f"  警告: 预测掩码 {pred_mask_npz_path} 形状 {pred_mask_binary_vis.shape} 不正确，应为 {(TARGET_SIZE, TARGET_SIZE, NUM_ORGANS)}")
                         pred_mask_binary_vis = None # 设为None，后续会处理
                except Exception as e_load_pred:
                    print(f"  加载预测掩码 {pred_mask_npz_path} 失败: {e_load_pred}")
            else:
                print(f"  未找到预测文件: {pred_mask_npz_path}。尝试实时预测...")
                if 'best_model' in locals() and best_model is not None:
                    try:
                        processed_image_for_pred = preprocess_image_for_unet(image_vis_raw, TARGET_SIZE)
                        prediction_prob_live = best_model.predict(np.expand_dims(processed_image_for_pred, axis=0), verbose=0)[0]
                        pred_mask_binary_vis = (prediction_prob_live > PREDICTION_THRESHOLD).astype(np.uint8)
                    except Exception as live_pred_e:
                        print(f"    实时预测失败: {live_pred_e}")
                else:
                    print("    best_model 未定义或为None，无法进行实时预测。")

            # --- 创建彩色叠加图 ---
            display_image_resized = cv2.resize(image_vis_raw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
            display_image_rgb = cv2.cvtColor((display_image_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # 真实掩码叠加
            gt_overlay_vis = np.zeros_like(display_image_rgb, dtype=np.uint8)
            for nii_val_map, org_name_map in ORGAN_MAP_NII.items(): # 使用全局变量
                 if org_name_map in colors_for_visualization: # 使用新定义的颜色字典
                     color_bgr_val = colors_for_visualization[org_name_map]
                     current_gt_mask_channel_raw = np.zeros(gt_slice_int_oriented_raw_res.shape, dtype=np.uint8)
                     if org_name_map == 'kidney':
                         current_gt_mask_channel_raw = ((gt_slice_int_oriented_raw_res == 3) | (gt_slice_int_oriented_raw_res == 4)).astype(np.uint8)
                     else:
                         current_gt_mask_channel_raw = (gt_slice_int_oriented_raw_res == nii_val_map).astype(np.uint8)
                     
                     current_gt_mask_channel_resized = cv2.resize(current_gt_mask_channel_raw, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
                     for c_idx_rgb in range(3):
                        gt_overlay_vis[current_gt_mask_channel_resized > 0, c_idx_rgb] = color_bgr_val[c_idx_rgb]
            
            alpha_blend = 0.4 # 透明度
            gt_blended_vis = cv2.addWeighted(display_image_rgb, 1 - alpha_blend, gt_overlay_vis, alpha_blend, 0)

            # 预测掩码叠加
            pred_blended_vis = display_image_rgb.copy() # 默认为原始图像
            if pred_mask_binary_vis is not None:
                pred_overlay_vis = np.zeros_like(display_image_rgb, dtype=np.uint8)
                for org_name_pred, channel_idx_pred in ORGAN_CHANNEL_MAP.items(): # 使用全局变量
                    if org_name_pred in colors_for_visualization and channel_idx_pred < pred_mask_binary_vis.shape[2]:
                        mask_ch_pred_display = pred_mask_binary_vis[:, :, channel_idx_pred]
                        color_bgr_pred_val = colors_for_visualization[org_name_pred]
                        for c_idx_rgb_pred in range(3):
                            pred_overlay_vis[mask_ch_pred_display > 0, c_idx_rgb_pred] = color_bgr_pred_val[c_idx_rgb_pred]
                pred_blended_vis = cv2.addWeighted(display_image_rgb, 1 - alpha_blend, pred_overlay_vis, alpha_blend, 0)
            else:
                print(f"  患者 {patient_id_vis} 切片 {inst_num_vis} 无有效预测掩码用于叠加。")


            # --- 显示 ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f"患者 {patient_id_vis} - DICOM索引 {dicom_list_idx_vis} (Inst: {inst_num_vis}) - NII索引 {nii_slice_idx_for_gt}", fontsize=16)

            axes[0].imshow(cv2.cvtColor(display_image_rgb, cv2.COLOR_BGR2RGB))
            axes[0].set_title("原始DICOM图像 (调整大小后)")
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(gt_blended_vis, cv2.COLOR_BGR2RGB))
            axes[1].set_title("真实掩码叠加")
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(pred_blended_vis, cv2.COLOR_BGR2RGB))
            axes[2].set_title("预测掩码叠加")
            axes[2].axis('off')
            
            # 添加图例
            legend_elements = [plt.Rectangle((0, 0), 1, 1, color=[c/255. for c in colors_for_visualization[org][::-1]], label=org) # Matplotlib 用 RGB 0-1
                               for org in ORGAN_CHANNEL_MAP.keys() if org in colors_for_visualization]
            fig.legend(handles=legend_elements, loc='lower center', ncol=len(ORGAN_CHANNEL_MAP.keys()), bbox_to_anchor=(0.5, -0.02))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整以防标题和图例重叠
            plt.savefig(os.path.join(OUTPUT_DIR, f"vis_pred_gt_{patient_id_vis}_dcm{dicom_list_idx_vis}_nii{nii_slice_idx_for_gt}.png"))
            plt.show()
            
            gc.collect() # 清理内存
            
    print("-" * 30)
    print("可视化部分执行完毕。")
    print("-" * 30)

if __name__ == "__main__":
    main()

