# -*- coding: utf-8 -*-
import gc
import os
import numpy as np
import pandas as pd
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import random
import glob
import nibabel as nib
import math
import json
import time
import shutil
import traceback  # 添加此库来获取详细的错误信息
import psutil  # 添加此库来监控内存使用情况
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import cv2

# === 基本配置 ===
# --- 数据路径 ---
DATA_DIR = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_images')
SEGMENTATION_DIR = os.path.join(DATA_DIR, 'segmentations')
META_DIR = DATA_DIR # Assuming meta files are in the root DATA_DIR

# --- 输出路径 (在 Kaggle 工作目录中) ---
PREPROCESSED_DIR = '/kaggle/working/preprocessed_data'
# 为了能够在笔记本重启后恢复进度，我们需要使用Kaggle的输出目录
OUTPUT_DATASET_DIR = '/kaggle/output/preprocessed-segmentation-data'

# 确保目录存在
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

# --- 图像和模型参数 ---
IMG_SIZE = (224, 224) # 你可以选择 128, 192, 224 等
TARGET_SIZE_INT = IMG_SIZE[0]
TARGET_SIZE = IMG_SIZE[0]
N_INPUT_CHANNELS = 3 # 如果你的模型需要 3 通道输入
BEST_NIFTI_ORIENTATION_TRANSFORM = ['rot90'] # 根据您的调试结果
USE_REVERSE_NIFTI_MAPPING = True # 设置为True来启用反向映射

# --- 器官映射 ---
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
NUM_ORGANS = len(ORGAN_CHANNEL_MAP)

# --- 并行处理 ---
# 减少工作进程数，避免内存问题
NUM_WORKERS = 2  # 从4减少到2
# 批处理参数
PATIENT_BATCH_SIZE = 5  # 从20减少到5
SLICE_BATCH_SIZE = 10   # 从20减少到10
SAVE_INTERVAL = 5       # 从10减少到5，更频繁保存进度

# --- 日志和进度文件 ---
LOG_FILE = os.path.join(PREPROCESSED_DIR, 'preprocessing_log.txt')

# --- 添加内存监控阈值 ---
MEMORY_THRESHOLD = 85  # 如果内存使用超过85%，暂停处理

# === 辅助函数 ===
def log_message(message):
    """记录消息到日志文件和控制台"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    print(log_line)
    
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

def get_memory_usage():
    """获取当前内存使用百分比"""
    return psutil.virtual_memory().percent

def check_memory():
    """检查内存使用情况，如果超过阈值则等待"""
    mem_usage = get_memory_usage()
    if mem_usage > MEMORY_THRESHOLD:
        log_message(f"内存使用率高: {mem_usage}%，超过阈值 {MEMORY_THRESHOLD}%，暂停处理并强制GC")
        gc.collect()
        time.sleep(5)  # 等待5秒，让系统有时间释放内存
        return False
    return True

# ... (log_message, get_memory_usage, check_memory 等函数之后) ...

def apply_orientation_transform(mask_slice, transform_ops=None):
    """
    应用方向变换到掩码切片。
    transform_ops: 一个包含操作字符串的列表，例如 ['rot90', 'fliplr']
                   可能的op: 'rot90', 'rot90_2', 'rot90_3', 'fliplr', 'flipud'
    """
    if transform_ops is None:
        return mask_slice

    transformed_mask = mask_slice.copy() #确保在副本上操作
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
            log_message(f"警告: 未知的方向变换操作 '{op}'") # 使用log_message
    return transformed_mask

def backup_progress_files():
    """备份进度文件到输出数据集目录"""
    # 备份全局进度文件
    global_progress_file = os.path.join(PREPROCESSED_DIR, 'global_progress.json')
    if os.path.exists(global_progress_file):
        shutil.copy(global_progress_file, os.path.join(OUTPUT_DATASET_DIR, 'global_progress.json'))
    
    # 备份日志文件
    if os.path.exists(LOG_FILE):
        shutil.copy(LOG_FILE, os.path.join(OUTPUT_DATASET_DIR, 'preprocessing_log.txt'))
    
    log_message("进度文件已备份到输出数据集目录")

def restore_progress_files():
    """从输出数据集目录恢复进度文件"""
    # 恢复全局进度文件
    output_global_progress = os.path.join(OUTPUT_DATASET_DIR, 'global_progress.json')
    if os.path.exists(output_global_progress):
        shutil.copy(output_global_progress, os.path.join(PREPROCESSED_DIR, 'global_progress.json'))
        log_message("已从输出数据集目录恢复全局进度文件")
    
    # 恢复日志文件
    output_log = os.path.join(OUTPUT_DATASET_DIR, 'preprocessing_log.txt')
    if os.path.exists(output_log):
        # 追加模式，保留当前日志
        with open(output_log, 'r') as src, open(LOG_FILE, 'a') as dst:
            dst.write(src.read())
        log_message("已从输出数据集目录恢复日志文件")

def load_dicom_slice(path):
    try:
        dicom_file = pydicom.dcmread(path)
        instance_number = int(dicom_file.InstanceNumber)
        image = apply_voi_lut(dicom_file.pixel_array, dicom_file)

        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)

        image = image.astype(np.float32)

        if 'PhotometricInterpretation' in dicom_file and dicom_file.PhotometricInterpretation == "MONOCHROME1":
             image = 1.0 - image

        return image, instance_number
    except Exception as e:
        # 增加更详细的错误日志
        log_message(f"加载 DICOM 错误 {path}: {e}")
        return None, None

def load_multi_organ_segmentation_mask(nii_data_array, slice_index, 
                                       organ_map_nii, organ_channel_map_local, # 避免与全局变量冲突
                                       num_organs_local, target_size_local_int, # 使用局部变量名
                                       orientation_transform_ops_local=None,
                                       nii_path_for_error_msg=None): # 新增参数
    try:
        seg_data = nii_data_array
        if not isinstance(seg_data, np.ndarray) or seg_data.ndim != 3:
             log_message(f"NII数据无效: 不是3D数组，形状: {seg_data.shape if hasattr(seg_data, 'shape') else '未知'}")
             return None

        num_slices = seg_data.shape[2]
        if not (0 <= slice_index < num_slices):
            # log_message(f"切片索引 {slice_index} 超出范围 (0-{num_slices-1})") # 此日志可以在调用处处理
            return None

        mask_slice_float = seg_data[:, :, slice_index]

        # ***** 应用方向变换 *****
        if orientation_transform_ops_local:
            mask_slice_float = apply_orientation_transform(mask_slice_float, orientation_transform_ops_local)
        # ***********************
            
        mask_slice = np.round(mask_slice_float).astype(np.int16)

        # multi_channel_mask 的尺寸基于变换后的 mask_slice
        multi_channel_mask = np.zeros((mask_slice.shape[0], mask_slice.shape[1], num_organs_local), dtype=np.float32)

        for nii_value, organ_name in organ_map_nii.items():
            if organ_name in organ_channel_map_local:
                channel_idx = organ_channel_map_local[organ_name]
                if organ_name == 'kidney':
                    binary_mask_organ = ((mask_slice == 3) | (mask_slice == 4)).astype(np.float32)
                else:
                    binary_mask_organ = (mask_slice == nii_value).astype(np.float32)
                
                # 确保维度匹配后才相加
                if binary_mask_organ.shape == multi_channel_mask.shape[:2]:
                    multi_channel_mask[:, :, channel_idx] += binary_mask_organ
                else:
                    log_message(f"警告: 器官 {organ_name} 的二值掩码形状 {binary_mask_organ.shape} 与基底 {multi_channel_mask.shape[:2]} 不匹配，尝试resize。")
                    resized_bmo = cv2.resize(binary_mask_organ, (multi_channel_mask.shape[1], multi_channel_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    multi_channel_mask[:, :, channel_idx] += resized_bmo


        # 现在 multi_channel_mask 是在（可能经过变换的）原始NII切片分辨率下的
        # 将其resize到目标尺寸
        if multi_channel_mask.shape[0] != target_size_local_int or multi_channel_mask.shape[1] != target_size_local_int:
            resized_mask = cv2.resize(
                multi_channel_mask,
                (target_size_local_int, target_size_local_int),
                interpolation=cv2.INTER_NEAREST
            )
            if len(resized_mask.shape) == 2 and num_organs_local == 1:
                 resized_mask = np.expand_dims(resized_mask, axis=-1)
            elif len(resized_mask.shape) == 2 and num_organs_local > 1:
                 log_message(f"调整大小后的掩码形状错误: {resized_mask.shape} (应为3D)，用于切片索引 {slice_index}")
                 return None # 多通道压缩通常是错误
        else:
             resized_mask = multi_channel_mask # 如果尺寸已匹配，则无需resize
        
        final_mask = (resized_mask > 0.5).astype(np.float32) # 确保是 0 或 1

        if final_mask.shape != (target_size_local_int, target_size_local_int, num_organs_local):
             log_message(f"最终掩码形状错误: {final_mask.shape}, 应为 ({target_size_local_int}, {target_size_local_int}, {num_organs_local}) 用于切片索引 {slice_index}")
             return None

        return final_mask

    except Exception as e:
        log_message(f"生成掩码时出错 (切片索引 {slice_index}): {e}")
        # log_message(f"详细错误: {traceback.format_exc()}") # 可选，用于更详细调试
        return None
  

def preprocess_image_for_unet(image, target_size):
    try:
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        if N_INPUT_CHANNELS == 3:
            image_rgb = np.stack([image_resized] * 3, axis=-1)
            return image_rgb.astype(np.float32)
        else: # Assuming 1 channel if not 3
            return np.expand_dims(image_resized, axis=-1).astype(np.float32)
    except Exception as e:
        log_message(f"预处理图像时出错: {e}")
        return None

def get_dicom_files_dict(patient_id, series_id, dicom_tags_df):
    sorted_dicom_info = []
    patient_dir = os.path.join(TRAIN_IMAGES_DIR, str(patient_id))
    series_folder = os.path.join(patient_dir, str(series_id))
    
    try:
        dicom_files = glob.glob(os.path.join(series_folder, '*.dcm'))
        if not dicom_files: 
            log_message(f"患者 {patient_id} 系列 {series_id} 没有找到DICOM文件")
            return []

        dicom_tuples = []
        use_tags = False
        # 尝试使用 DICOM tags 排序 (如果提供了 tags 文件)
        if dicom_tags_df is not None and all(col in dicom_tags_df.columns for col in ['PatientID', 'SeriesInstanceUID', 'InstanceNumber', 'SOPInstanceUID', 'series_id_extracted']):
            try:
                patient_id_str = str(patient_id)
                tags_subset = dicom_tags_df[
                     (dicom_tags_df['PatientID'].astype(str) == patient_id_str) &
                     (dicom_tags_df['series_id_extracted'] == str(series_id)) # 使用提取的 series_id
                ][['InstanceNumber', 'SOPInstanceUID']].dropna()

                if not tags_subset.empty:
                    sop_to_inst = dict(zip(tags_subset['SOPInstanceUID'], tags_subset['InstanceNumber'].astype(int)))
                    use_tags = True
                    for f_path in dicom_files:
                        try:
                            # 只读头，不读像素数据，更快
                            ds_header = pydicom.dcmread(f_path, stop_before_pixels=True)
                            ds_sop = ds_header.SOPInstanceUID
                            if ds_sop in sop_to_inst:
                                 dicom_tuples.append((sop_to_inst[ds_sop], f_path))
                            else: # Fallback: read InstanceNumber directly from header
                                 dicom_tuples.append((int(ds_header.InstanceNumber), f_path))
                        except Exception as e:
                            log_message(f"读取DICOM头失败 {f_path}: {e}")
            except Exception as e:
                log_message(f"使用DICOM tags排序失败: {e}")
                use_tags = False

        # 如果 Tags 失败或不可用，尝试直接从 DICOM 头读取 InstanceNumber
        if not use_tags or not dicom_tuples:
            dicom_tuples = []
            for f_path in dicom_files:
                try:
                    ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                    dicom_tuples.append((int(ds.InstanceNumber), f_path))
                except Exception as e:
                    log_message(f"读取DICOM头失败 {f_path}: {e}")

        # 如果 DICOM 头也失败，按文件名中的数字排序
        if not dicom_tuples:
            try:
               dicom_tuples = sorted([(int(os.path.splitext(os.path.basename(f))[0]), f) for f in dicom_files])
            except ValueError:
               dicom_tuples = sorted([(i, f) for i, f in enumerate(sorted(dicom_files))]) # 按字母顺序

        dicom_tuples.sort(key=lambda x: x[0])
        sorted_dicom_info = [(item[0], item[1]) for item in dicom_tuples] # (InstanceNumber, path)
        
        return sorted_dicom_info
    except Exception as e:
        log_message(f"获取DICOM文件列表失败 (患者 {patient_id}, 系列 {series_id}): {e}")
        return []

def copy_patient_data_to_output(patient_id):
    """将患者的预处理数据复制到输出数据集目录"""
    try:
        src_dir = os.path.join(PREPROCESSED_DIR, str(patient_id))
        dst_dir = os.path.join(OUTPUT_DATASET_DIR, str(patient_id))
        
        if os.path.exists(src_dir):
            # 确保目标目录存在
            os.makedirs(dst_dir, exist_ok=True)
            
            # 复制所有文件
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            
            return True
    except Exception as e:
        log_message(f"复制患者数据到输出目录失败 (患者 {patient_id}): {e}")
        return False

# === 预处理核心函数 ===
# 在 === 预处理核心函数 === 部分
def preprocess_and_save_data(args):
    patient_id, dicom_info_list, nii_path, output_dir_base, target_size_int = args # 修改了dicom_paths为dicom_info_list
    
    # dicom_info_list 是 [(instance_number, dicom_path), ...] 的列表，已经排序

    try:
        log_message(f"开始处理患者 {patient_id}...")
        patient_output_dir = os.path.join(output_dir_base, str(patient_id)) # 使用 output_dir_base
        os.makedirs(patient_output_dir, exist_ok=True)
        
        progress_file = os.path.join(patient_output_dir, 'progress.json')
        # ... (进度文件加载逻辑不变) ...
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    if progress.get('completed', False):
                        log_message(f"患者 {patient_id} 已经处理完成 (根据进度文件)，跳过。")
                        return progress.get('processed_count', 0), patient_id
                    processed_instances = set(map(str, progress.get('processed_instances', []))) # 确保是字符串集合
                    log_message(f"患者 {patient_id} 恢复进度: 已处理 {len(processed_instances)} 个切片。")
            except Exception as e:
                log_message(f"读取患者 {patient_id} 的进度文件 {progress_file} 失败: {e}。重新开始处理该患者。")
                processed_instances = set()
        else:
            processed_instances = set()

        try:
            # log_message(f"尝试加载患者 {patient_id} 的NII文件 {nii_path}")
            nii_img = nib.load(nii_path)
            # log_message(f"成功加载NII文件头部，准备读取数据")
            nii_data = nii_img.get_fdata(dtype=np.float32)
            # log_message(f"成功加载NII数据，形状: {nii_data.shape}")
        except Exception as e:
            log_message(f"!!! 错误 - 患者 {patient_id}: 无法加载NII文件 {nii_path}: {e}")
            # log_message(f"详细错误: {traceback.format_exc()}")
            return 0, patient_id

        actual_processed_count = 0 # 本次运行实际处理的切片数
        nii_total_slices = nii_data.shape[2]
        num_dicom_files = len(dicom_info_list)
        
        # log_message(f"患者 {patient_id} 共有 {num_dicom_files} 个DICOM文件和 {nii_total_slices} 个NII切片")
        if num_dicom_files == 0:
            log_message(f"患者 {patient_id}: DICOM文件列表为空，跳过。")
            return 0, patient_id

        for dicom_list_idx, (instance_number, dicom_path) in enumerate(dicom_info_list):
            if str(instance_number) in processed_instances: # 检查是否已处理
                continue

            # ***** 核心修改：应用反向索引映射 *****
            if USE_REVERSE_NIFTI_MAPPING: # 使用全局配置
                nii_slice_idx_to_use = nii_total_slices - 1 - dicom_list_idx
            else: # 直接映射 (如果以后想改回来)
                nii_slice_idx_to_use = dicom_list_idx
            # *****************************************

            if not (0 <= nii_slice_idx_to_use < nii_total_slices):
                # log_message(f"患者 {patient_id}: 为DICOM索引 {dicom_list_idx} (Inst: {instance_number}) 计算的NIFTI索引 {nii_slice_idx_to_use} 超出范围 [0, {nii_total_slices-1}]。跳过。")
                continue
            
            # 分批逻辑似乎在 perform_offline_preprocessing 中，这里是单个 slice
            # 检查内存可以放在外层循环（例如 perform_offline_preprocessing 中每处理一个病人后）
            # 或者如果单个切片处理也可能导致问题，可以保留，但可能过于频繁

            image_data, _ = load_dicom_slice(dicom_path) # instance_number 已从 dicom_info_list 获取
            if image_data is None:
                continue
            
            mask_data = load_multi_organ_segmentation_mask(
                nii_data,
                nii_slice_idx_to_use,
                ORGAN_MAP_NII,          # 全局
                ORGAN_CHANNEL_MAP,      # 全局
                NUM_ORGANS,             # 全局
                target_size_int,        # 确保是整数
                nii_path_for_error_msg=f"Pat_{patient_id}_DcmIdx_{dicom_list_idx}_NiiIdx_{nii_slice_idx_to_use}", # 传递信息
                orientation_transform_ops_local=BEST_NIFTI_ORIENTATION_TRANSFORM # 使用全局定义的变换
            )

            if mask_data is None:
                continue
            
            processed_image_data = preprocess_image_for_unet(image_data, target_size_int)
            if processed_image_data is None:
                continue

            output_filename = f"{instance_number}.npz" # 使用 InstanceNumber 作为文件名
            output_file_path = os.path.join(patient_output_dir, output_filename)

            try:
                np.savez_compressed(output_file_path, image=processed_image_data, mask=mask_data)
                actual_processed_count += 1
                processed_instances.add(str(instance_number)) # 添加到已处理集合

                if actual_processed_count > 0 and actual_processed_count % (SAVE_INTERVAL * 5) == 0: # 减少打印频率
                    log_message(f"患者 {patient_id}: 已处理 {actual_processed_count} 个新切片...")
            except Exception as e_save_npz:
                log_message(f"保存NPZ文件失败 {output_file_path}: {e_save_npz}")
        
        # 患者所有切片处理（或尝试处理）完毕后，更新并保存该患者的进度文件
        final_processed_this_run = len(processed_instances)
        with open(progress_file, 'w') as f:
            json.dump({
                'processed_instances': list(processed_instances),
                'processed_count': final_processed_this_run, # 总共处理的实例数
                'completed': True, # 标记此患者处理尝试已完成
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f)
        
        log_message(f"患者 {patient_id} 处理完成。本次运行新处理 {actual_processed_count} 个切片。总计已处理 {final_processed_this_run} 个切片。")
        
        # 显式释放内存
        del nii_data, nii_img
        gc.collect()
        
        return final_processed_this_run, patient_id # 返回的是该患者总共完成的切片数

    except Exception as e_outer:
        log_message(f"处理患者 {patient_id} 时发生顶层异常: {e_outer}")
        log_message(f"详细错误: {traceback.format_exc()}")
        return 0, patient_id # 返回处理失败


def perform_offline_preprocessing(image_paths_map, segmentation_map, output_dir, target_size):
    """分批处理患者数据并保存进度"""
    log_message(f"开始离线预处理数据，将使用序列处理替代并行处理...")
    start_time = time.time()
    
    # 尝试从输出数据集目录恢复进度文件
    restore_progress_files()

    # 检查全局进度文件
    global_progress_file = os.path.join(output_dir, 'global_progress.json')
    if os.path.exists(global_progress_file):
        try:
            with open(global_progress_file, 'r') as f:
                global_progress = json.load(f)
                processed_patient_ids = set(global_progress.get('processed_patient_ids', []))
                log_message(f"加载全局进度：已处理 {len(processed_patient_ids)} 位患者")
        except Exception as e:
            log_message(f"读取全局进度文件失败: {e}")
            processed_patient_ids = set()
    else:
        processed_patient_ids = set()

    # 获取未处理的患者列表
    patient_ids_to_process = [pid for pid in sorted(list(image_paths_map.keys())) 
                         if pid in segmentation_map and pid not in processed_patient_ids]
    
    log_message(f"需要处理 {len(patient_ids_to_process)} 位患者")
    
    # 分批处理患者，每批PATIENT_BATCH_SIZE个
    for batch_idx in range(0, len(patient_ids_to_process), PATIENT_BATCH_SIZE):
        batch_end = min(batch_idx + PATIENT_BATCH_SIZE, len(patient_ids_to_process))
        current_batch = patient_ids_to_process[batch_idx:batch_end]
        
        log_message(f"\n处理批次 {batch_idx//PATIENT_BATCH_SIZE + 1}/{math.ceil(len(patient_ids_to_process)/PATIENT_BATCH_SIZE)}, "
              f"患者 {batch_idx}-{batch_end-1}")
        
        tasks = []
        for patient_id in current_batch:
            task_args = (
                patient_id,
                image_paths_map[patient_id],
                segmentation_map[patient_id],
                output_dir,
                target_size
            )
            tasks.append(task_args)

        batch_processed_slices = 0
        batch_processed_patients = []
        batch_failed_patients = []

        # 使用序列处理代替ProcessPoolExecutor
        log_message("使用序列处理替代并行处理...")
        for task_idx, task in enumerate(tasks):
            patient_id = task[0]
            log_message(f"开始处理患者 {task_idx+1}/{len(tasks)}: {patient_id}")
            
            # 检查内存使用情况
            if not check_memory():
                log_message(f"内存使用过高，跳过患者 {patient_id}")
                batch_failed_patients.append(patient_id)
                continue
                
            try:
                processed_count, patient_id = preprocess_and_save_data(task)
                if processed_count > 0:
                    batch_processed_slices += processed_count
                    batch_processed_patients.append(patient_id)
                    processed_patient_ids.add(patient_id)
                    
                    # 将处理好的患者数据复制到输出数据集目录
                    copy_patient_data_to_output(patient_id)
                    log_message(f"成功处理患者 {patient_id}，共 {processed_count} 个切片")
                else:
                    batch_failed_patients.append(patient_id)
                    log_message(f"处理患者 {patient_id} 失败，返回切片数为0")
            except Exception as e:
                log_message(f"处理患者 {patient_id} 时发生异常: {e}")
                log_message(f"详细错误: {traceback.format_exc()}")
                batch_failed_patients.append(patient_id)
            
            # 每个患者处理完后立即清理内存
            gc.collect()
            log_message(f"已完成 {task_idx+1}/{len(tasks)} 个患者")
        
        # 每批处理完后更新全局进度并备份
        with open(global_progress_file, 'w') as f:
            json.dump({
                'processed_patient_ids': list(processed_patient_ids),
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_patients': len(image_paths_map),
                'failed_patients': batch_failed_patients
            }, f)
        
        # 备份进度文件
        backup_progress_files()
        
        log_message(f"批次 {batch_idx//PATIENT_BATCH_SIZE + 1} 完成，处理了 {len(batch_processed_patients)} 位患者，"
              f"{batch_processed_slices} 个切片")
        
        # 清理内存
        gc.collect()

    end_time = time.time()
    log_message(f"\n预处理完成。")
    log_message(f"  耗时: {end_time - start_time:.2f} 秒")
    log_message(f"  成功处理患者数: {len(processed_patient_ids)} / {len(image_paths_map)}")
    log_message(f"  数据保存在: {output_dir} 和 {OUTPUT_DATASET_DIR}")

    return list(processed_patient_ids)
# 在 === 辅助函数 === 部分或主流程中
def visualize_alignment_check(patient_id, slice_idx_in_dicom_list, # 改为DICOM列表中的索引
                              dicom_info_list, nii_path, target_display_size_int): # target_size -> target_display_size_int
    
    if not (0 <= slice_idx_in_dicom_list < len(dicom_info_list)):
        log_message(f"错误: DICOM列表索引 {slice_idx_in_dicom_list} 超出范围 (患者 {patient_id})")
        return

    instance_number, dicom_path = dicom_info_list[slice_idx_in_dicom_list]

    log_message(f"可视化对齐检查 - 患者: {patient_id}, DICOM列表索引: {slice_idx_in_dicom_list}, InstanceNum: {instance_number}")

    try:
        dicom_file = pydicom.dcmread(dicom_path)
        image_orig = apply_voi_lut(dicom_file.pixel_array, dicom_file)
        img_min, img_max = np.min(image_orig), np.max(image_orig)
        image_display = (image_orig - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(image_orig)
        if 'PhotometricInterpretation' in dicom_file and dicom_file.PhotometricInterpretation == "MONOCHROME1":
            image_display = 1.0 - image_display
    except Exception as e:
        log_message(f"  加载DICOM {dicom_path} 错误: {e}")
        return

    try:
        nii_img = nib.load(nii_path)
        nii_data = nii_img.get_fdata(dtype=np.float32)
        nii_total_slices = nii_data.shape[2]

        # ***** 核心修改：应用反向索引映射和方向变换 *****
        nii_slice_idx_to_use = nii_total_slices - 1 - slice_idx_in_dicom_list # 反向映射
        
        if not (0 <= nii_slice_idx_to_use < nii_total_slices):
            log_message(f"  错误: 为DICOM索引 {slice_idx_in_dicom_list} 计算的NIFTI索引 {nii_slice_idx_to_use} 超出范围 ({nii_total_slices}片)。")
            return

        mask_slice_raw = nii_data[:, :, nii_slice_idx_to_use]
        mask_slice_oriented = apply_orientation_transform(mask_slice_raw, BEST_NIFTI_ORIENTATION_TRANSFORM) # 全局
        mask_slice_int = np.round(mask_slice_oriented).astype(np.int16)
        # ****************************************************

        colors_rgb = { # Matplotlib 使用 RGB
            'liver': [1, 0, 0],  # Red
            'spleen': [0, 1, 0],  # Green
            'kidney': [0, 0, 1],  # Blue
            'bowel': [1, 1, 0]   # Yellow
        }
        original_mask_shape = mask_slice_int.shape
        mask_overlay_rgb = np.zeros((original_mask_shape[0], original_mask_shape[1], 3), dtype=np.float32) # Use float for overlay

        for nii_val, organ_name in ORGAN_MAP_NII.items(): # 全局
             if organ_name in ORGAN_CHANNEL_MAP: # 确保是我们关心的器官
                color_to_use = colors_rgb.get(organ_name, [0.5, 0.5, 0.5]) # 默认灰色
                if organ_name == 'kidney':
                     current_mask_bool = ((mask_slice_int == 3) | (mask_slice_int == 4))
                else:
                     current_mask_bool = (mask_slice_int == nii_val)
                
                for c in range(3):
                    mask_overlay_rgb[current_mask_bool, c] = color_to_use[c]
        
    except Exception as e:
        log_message(f"  加载NII或生成掩码时出错 (NII路径 {nii_path}, NII索引 {nii_slice_idx_to_use}): {e}")
        # log_message(f"  详细错误: {traceback.format_exc()}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 调整DICOM和掩码以匹配显示 (例如，都resize到target_display_size_int)
    image_display_resized = cv2.resize(image_display, (target_display_size_int, target_display_size_int), interpolation=cv2.INTER_LINEAR)
    mask_overlay_rgb_resized = cv2.resize(mask_overlay_rgb, (target_display_size_int, target_display_size_int), interpolation=cv2.INTER_NEAREST)

    axes[0].imshow(image_display_resized, cmap='gray')
    axes[0].set_title(f"DICOM (Idx:{slice_idx_in_dicom_list}, Inst:{instance_number})")
    axes[0].axis('off')

    axes[1].imshow(mask_overlay_rgb_resized) # mask_overlay_rgb_resized 已经是RGB float
    axes[1].set_title(f"NII Mask (NII Idx:{nii_slice_idx_to_use}, Transform:{BEST_NIFTI_ORIENTATION_TRANSFORM})")
    axes[1].axis('off')
    
    image_display_rgb_resized = cv2.cvtColor((image_display_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # alpha blend: image_display_rgb_resized 和 mask_overlay_rgb_resized (确保mask_overlay_rgb_resized也是0-1范围或通过alpha混合)
    # 确保 mask_overlay_rgb_resized 的非零部分才参与混合
    mask_for_blending = (np.sum(mask_overlay_rgb_resized, axis=2) > 0).astype(np.uint8) # 找到有颜色的区域
    
    blended_image = image_display_rgb_resized.copy()
    # 只在有掩码的地方进行混合
    alpha = 0.4
    for r_idx in range(image_display_rgb_resized.shape[0]):
        for c_idx in range(image_display_rgb_resized.shape[1]):
            if mask_for_blending[r_idx, c_idx] > 0: # 如果掩码在此处有颜色
                for channel in range(3):
                    blended_image[r_idx, c_idx, channel] = \
                        (1 - alpha) * image_display_rgb_resized[r_idx, c_idx, channel] + \
                        alpha * (mask_overlay_rgb_resized[r_idx, c_idx, channel] * 255) # 假设mask_overlay是0-1 float
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)


    axes[2].imshow(blended_image)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    # Add legend (using ORGAN_CHANNEL_MAP keys for consistency with model output)
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=colors_rgb[org], label=org)
                       for org in ORGAN_CHANNEL_MAP.keys() if org in colors_rgb]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(ORGAN_CHANNEL_MAP.keys()), bbox_to_anchor=(0.5, -0.01))
    plt.suptitle(f"Alignment Check - Patient {patient_id}", fontsize=16) # y调整到1.0或略大
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 在主流程中调用 visualize_alignment_check 时 (Cell 12 的末尾 "--- 3. 执行可视化对齐检查 (抽样) ---" 部分)
# 确保传入的是 TARGET_SIZE_INT
# visualize_alignment_check(
#     patient_id,
#     slice_idx,
#     dicom_info_list,
#     nii_path,
#     TARGET_SIZE_INT # 使用整数尺寸
# )

# === 主执行流程 ===
if __name__ == "__main__":
    log_message("--- 开始预处理 Notebook ---")
    log_message(f"目标图像尺寸: {IMG_SIZE}")
    log_message(f"输出目录: {PREPROCESSED_DIR}")
    log_message(f"输出数据集目录: {OUTPUT_DATASET_DIR}")
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

    log_message("\n--- 1. 构建文件映射 ---")
    # 加载元数据
    train_meta_path = os.path.join(META_DIR, 'train_series_meta.csv')
    dicom_tags_path = os.path.join(META_DIR, 'train_dicom_tags.parquet') # parquet 文件路径

    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"找不到系列元数据文件: {train_meta_path}")
    
    try:
        log_message(f"加载训练系列元数据: {train_meta_path}")
        train_series_meta = pd.read_csv(train_meta_path)
        log_message(f"成功加载元数据，包含 {len(train_series_meta)} 条记录")
    except Exception as e:
        log_message(f"加载元数据失败: {e}")
        log_message(f"详细错误: {traceback.format_exc()}")
        raise SystemExit("无法继续预处理")

    # 创建 series_id -> patient_id 映射
    try:
        series_to_patient = dict(zip(
            train_series_meta['series_id'].astype(str),
            train_series_meta['patient_id'].astype(str)
        ))
        log_message(f"创建了 {len(series_to_patient)} 个系列ID到患者ID的映射")
    except Exception as e:
        log_message(f"创建系列ID映射失败: {e}")
        series_to_patient = {}

    # 创建 series_id -> nii_path 映射
    series_to_nii = {}
    try:
        log_message(f"查找分割文件目录: {SEGMENTATION_DIR}")
        segmentation_files = glob.glob(os.path.join(SEGMENTATION_DIR, "*.nii"))
        # 添加对.nii.gz文件的支持
        segmentation_files.extend(glob.glob(os.path.join(SEGMENTATION_DIR, "*.nii.gz")))
        log_message(f"发现 {len(segmentation_files)} 个 NII/NII.GZ 文件。")
        
        for fpath in segmentation_files:
            # NII 文件名通常是 series_id.nii 或 series_id.nii.gz
            if fpath.endswith('.nii.gz'):
                series_id = os.path.basename(fpath).replace('.nii.gz','')
            else:
                series_id = os.path.basename(fpath).replace('.nii','')
                
            if series_id in series_to_patient:
                series_to_nii[series_id] = fpath
                
        log_message(f"成功映射 {len(series_to_nii)} 个 NII 文件到有效的 series_id。")
    except Exception as e:
        log_message(f"创建NII文件映射失败: {e}")
        log_message(f"详细错误: {traceback.format_exc()}")

    # 加载 DICOM tags (如果可用)
    dicom_tags_df = None
    if os.path.exists(dicom_tags_path):
        log_message("加载 DICOM tags (parquet)...")
        try:
            dicom_tags_df = pd.read_parquet(dicom_tags_path)
            # 预处理 tags DataFrame (确保提取 series_id)
            if 'PatientID' in dicom_tags_df.columns:
                dicom_tags_df['PatientID'] = dicom_tags_df['PatientID'].astype(str)
            if 'SeriesInstanceUID' in dicom_tags_df.columns:
                # 提取 series_id (假设它在 UID 的倒数第二部分，根据实际情况调整)
                dicom_tags_df['series_id_extracted'] = dicom_tags_df['SeriesInstanceUID'].astype(str) # 假设 UID 就是 series_id
                dicom_tags_df = dicom_tags_df.dropna(subset=['series_id_extracted'])
            log_message(f"DICOM tags 加载完成，包含 {len(dicom_tags_df)} 条记录。")
        except Exception as e:
            log_message(f"加载 DICOM tags 失败: {e}. 将不使用 tags 进行排序。")
            log_message(f"详细错误: {traceback.format_exc()}")
            dicom_tags_df = None
    else:
        log_message("未找到 DICOM tags 文件，将仅依赖DICOM头或文件名排序。")

    log_message("\n--- 2. 关联图像和分割文件 ---")
    # {patient_id: [(inst_num1, path1), (inst_num2, path2), ...]}
    image_paths_map_full = {}
    # {patient_id: nii_path}
    segmentation_map_full = {}
    valid_patients_found = []

    try:
        all_patient_ids = sorted(train_series_meta['patient_id'].astype(str).unique())
        log_message(f"总共有 {len(all_patient_ids)} 个独特的患者 ID 在元数据中。")

        # 遍历所有在元数据中有记录的患者
        for patient_id in all_patient_ids:
            try:
                log_message(f"处理患者 {patient_id} 的数据...")
                patient_series_ids = train_series_meta[train_series_meta['patient_id'].astype(str) == patient_id]['series_id'].astype(str).tolist()

                if not patient_series_ids:
                    log_message(f"患者 {patient_id} 没有关联的系列ID，跳过")
                    continue

                found_valid_series_for_patient = False
                # 尝试为每个患者找到一个有效的 (图像存在 + NII 存在) 的序列
                for series_id in patient_series_ids:
                    series_img_path = os.path.join(TRAIN_IMAGES_DIR, patient_id, series_id)
                    nii_path = series_to_nii.get(series_id)

                    # 检查图像目录和NII文件是否都存在
                    if os.path.isdir(series_img_path) and nii_path:
                        log_message(f"找到患者 {patient_id} 的有效系列: {series_id}")
                        # 获取并排序该序列的 DICOM 文件信息 (InstanceNumber, Path)
                        dicom_info_list = get_dicom_files_dict(patient_id, series_id, dicom_tags_df)

                        if dicom_info_list: # 确保序列中有有效的 DICOM 文件
                            log_message(f"患者 {patient_id} 系列 {series_id} 有 {len(dicom_info_list)} 个DICOM文件")
                            image_paths_map_full[patient_id] = dicom_info_list
                            segmentation_map_full[patient_id] = nii_path
                            valid_patients_found.append(patient_id)
                            found_valid_series_for_patient = True
                            break # 每个患者只使用找到的第一个有效 series
                        else:
                            log_message(f"患者 {patient_id} 系列 {series_id} 没有有效的DICOM文件")
                    
                if not found_valid_series_for_patient:
                    log_message(f"患者 {patient_id} 没有找到有效的系列")
            except Exception as e:
                log_message(f"处理患者 {patient_id} 时出错: {e}")
                log_message(f"详细错误: {traceback.format_exc()}")
    except Exception as e:
        log_message(f"关联图像和分割文件时出错: {e}")
        log_message(f"详细错误: {traceback.format_exc()}")

    # 去重并排序最终的患者 ID 列表
    final_patient_ids = sorted(list(set(valid_patients_found)))
    log_message(f"\n成功映射了 {len(final_patient_ids)} 位患者的图像和对应分割文件。")

    if not final_patient_ids:
        raise SystemExit("错误：未能找到任何包含有效图像序列及对应NII文件的患者。停止执行。")

    # 过滤字典，只保留有效患者的数据
    image_paths_map_final = {pid: image_paths_map_full[pid] for pid in final_patient_ids}
    segmentation_map_final = {pid: segmentation_map_full[pid] for pid in final_patient_ids}

    log_message("\n--- 3. 执行可视化对齐检查 (抽样) ---")
    num_patients_to_check = 1  # 减少到1个以节省时间和内存
    num_slices_per_patient = 1  # 减少到1个以节省时间和内存

    if len(final_patient_ids) > 0:
        try:
            # Ensure we don't request more patients than available
            num_patients_to_check = min(num_patients_to_check, len(final_patient_ids))

            # Select random patients
            sample_patient_ids = random.sample(final_patient_ids, num_patients_to_check)

            for patient_id in sample_patient_ids:
                dicom_info_list = image_paths_map_final.get(patient_id)
                nii_path = segmentation_map_final.get(patient_id)

                if not dicom_info_list or not nii_path:
                    log_message(f"警告: 样本患者 {patient_id} 缺少 DICOM 或 NII 信息，跳过可视化。")
                    continue

                num_available_slices = len(dicom_info_list)
                if num_available_slices == 0:
                     log_message(f"警告: 样本患者 {patient_id} DICOM 列表为空，跳过可视化。")
                     continue

                # Ensure we don't request more slices than available
                num_slices_to_check_this_patient = min(num_slices_per_patient, num_available_slices)

                # Select random slice indices (from the list index 0 to n-1)
                # Avoid checking only the very first/last slices
                middle_slice_idx = num_available_slices // 2
                slice_indices_to_check = [middle_slice_idx]  # 只使用中间切片以简化

                for slice_idx in slice_indices_to_check:
                    try:
                        log_message(f"尝试可视化患者 {patient_id} 的切片 {slice_idx}")
                        visualize_alignment_check(
                            patient_id,
                            slice_idx,
                            dicom_info_list,
                            nii_path,
                            TARGET_SIZE_INT # Pass target size if needed by mask func internally (though visualization uses original)
                        )
                        log_message(f"成功可视化患者 {patient_id} 的切片 {slice_idx}")
                    except Exception as e:
                        log_message(f"可视化患者 {patient_id} 切片 {slice_idx} 时出错: {e}")
                        log_message(f"详细错误: {traceback.format_exc()}")
                
                log_message("-" * 30) # Separator between patients
                
                # 每个患者后清理内存
                gc.collect()
        except Exception as e:
            log_message(f"可视化对齐检查时出错: {e}")
            log_message(f"详细错误: {traceback.format_exc()}")
            log_message("继续执行预处理，跳过可视化")
    else:
        log_message("没有找到有效的患者进行可视化检查。")

    log_message("--- 可视化对齐检查完成 ---")

    # --- NOW START THE FULL PREPROCESSING ---
    log_message("\n--- 4. 执行离线数据预处理 ---")
    
    try:
        # 执行预处理并保存到工作目录和输出数据集目录
        processed_patients = perform_offline_preprocessing(
            image_paths_map_final,
            segmentation_map_final,
            PREPROCESSED_DIR,
            TARGET_SIZE
        )
        
        # 统计预处理结果
        log_message(f"\n预处理统计:")
        log_message(f"  总患者数: {len(final_patient_ids)}")
        log_message(f"  成功处理患者数: {len(processed_patients)}")
    except Exception as e:
        log_message(f"执行离线预处理时发生严重错误: {e}")
        log_message(f"详细错误: {traceback.format_exc()}")
        processed_patients = []
    
    # 确保所有数据都已复制到输出数据集目录
    log_message("\n--- 5. 确保数据已保存到输出数据集 ---")
    for patient_id in processed_patients:
        try:
            copy_success = copy_patient_data_to_output(patient_id)
            if copy_success:
                log_message(f"患者 {patient_id} 数据已成功复制到输出目录")
            else:
                log_message(f"警告: 患者 {patient_id} 数据复制失败")
        except Exception as e:
            log_message(f"复制患者 {patient_id} 数据时出错: {e}")
    
    # 最终备份进度文件
    try:
        backup_progress_files()
    except Exception as e:
        log_message(f"备份进度文件时出错: {e}")
    
    # 输出一些有用的统计信息
    try:
        total_slices = 0
        total_file_size = 0
        for patient_id in processed_patients:
            patient_dir = os.path.join(PREPROCESSED_DIR, str(patient_id))
            if os.path.exists(patient_dir):
                npz_files = glob.glob(os.path.join(patient_dir, "*.npz"))
                total_slices += len(npz_files)
                for npz_file in npz_files:
                    total_file_size += os.path.getsize(npz_file)
        
        log_message(f"  总切片数: {total_slices}")
        log_message(f"  总文件大小: {total_file_size / (1024*1024):.2f} MB")
    except Exception as e:
        log_message(f"计算统计信息时出错: {e}")
    
    # 如果使用Kaggle API，可以添加提交数据集的命令
    log_message("\n要将预处理数据作为数据集保存，请确保在笔记本设置中添加了输出数据集。")
    log_message("数据已保存到: " + OUTPUT_DATASET_DIR)

    log_message("\n--- 预处理 Notebook 执行完毕 ---")


