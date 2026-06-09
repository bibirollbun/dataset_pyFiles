!pip install dicomsdl


# ======================
# 基础配置
# ======================
DEBUG = False  # 调试模式开关（True时只处理前10个病例）
GLOBAL_WIDTH = 224  # 统一处理后的图像尺寸（224x224）

import os
import re
import glob
import time
import numpy as np
import pandas as pd
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
import dicomsdl as dicoml
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from IPython.display import HTML
from multiprocessing import Pool, cpu_count
import imageio

# 数据根目录
rd = '/kaggle/input/rsna-intracranial-aneurysm-detection'

# ======================
# 工具函数
# ======================
def atoi(text):
    """将字符串转为数字（用于自然排序）"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """自然排序键生成器"""
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def get_sort_key(path):
    """
    生成DICOM文件排序键：
    1. 优先使用InstanceNumber
    2. 其次使用ImagePositionPatient的Z轴坐标
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        instance_number = getattr(ds, 'InstanceNumber', None)
        image_position = getattr(ds, 'ImagePositionPatient', None)
        z_position = image_position[2] if image_position else None
        
        if instance_number is not None:
            return (int(instance_number), 0)
        elif z_position is not None:
            return (float('inf'), float(z_position))
        return (float('inf'), float('inf'))
    except:
        return (float('inf'), float('inf'))

# ======================
# DICOM排序处理
# ======================
def fast_sort_dicom_paths(dcm_paths):
    """快速排序DICOM文件路径"""
    sort_info = []
    for path in dcm_paths:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            instance_number = getattr(ds, 'InstanceNumber', None)
            position = getattr(ds, 'ImagePositionPatient', [None]*3)
            z = position[2] if position else None
            
            if instance_number is not None:
                sort_info.append((int(instance_number), 0, path))
            elif z is not None:
                sort_info.append((float('inf'), float(z), path))
            else:
                sort_info.append((float('inf'), float('inf'), path))
        except:
            sort_info.append((float('inf'), float('inf'), path))
    
    sort_info.sort()
    return [x[2] for x in sort_info]

def sort_series(args):
    """多进程排序的辅助函数"""
    series_uid, paths = args
    return series_uid, fast_sort_dicom_paths(paths)

# ======================
# DICOM图像处理
# ======================
def apply_dicom_windowing(img, window_center, window_width):
    """
    应用DICOM窗宽窗位处理
    :param img: 原始图像数组
    :param window_center: 窗位
    :param window_width: 窗宽
    :return: 处理后的8bit图像
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality):
    """获取不同模态的默认窗宽窗位"""
    windows = {
        'CT': (40, 80),      # 常规CT
        'CTA': (50, 350),    # CT血管造影
        'MRA': (600, 1200),  # MR血管造影
        'MRI': (40, 80),     # 常规MRI
    }
    return windows.get(modality, (40, 80))

def dicom_to_png(src_path, dst_path, width=224, to_rgb=False, apply_windowing=False, modality='CT'):
    """
    DICOM转PNG核心函数
    :param src_path: DICOM源路径
    :param dst_path: PNG输出路径
    :param width: 输出图像宽度
    :param to_rgb: 是否转为RGB
    :param apply_windowing: 是否应用窗宽窗位
    :param modality: 影像模态类型
    """
    try:
        # 读取DICOM文件
        dicom = pydicom.dcmread(src_path, force=True)
        if 'PixelData' not in dicom:
            print(f"[SKIP] 无像素数据: {src_path}")
            return

        # 获取像素数组
        img = dicom.pixel_array
        interp = dicom.PhotometricInterpretation

        # 处理YBR色彩空间
        if interp == "YBR_FULL":
            img = convert_color_space(img, 'YBR_FULL', 'RGB')

        # 转为灰度图
        if img.ndim == 3:
            if interp in ["RGB", "YBR_FULL"]:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] > 3:
                img = img[:, :, 0]

        # 应用窗宽窗位
        if apply_windowing:
            window_center, window_width = get_windowing_params(modality)
            img = apply_dicom_windowing(img, window_center, window_width)

        # 归一化处理
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        else:
            img[:] = 0

        # 处理MONOCHROME1（低值=白色）
        if interp == "MONOCHROME1":
            img = 1 - img

        # 转为8bit并调整尺寸
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (width, width))

        # 转为RGB（如果需要）
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # 保存图像
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, img)

    except Exception as e:
        print(f"[ERROR] 处理失败 {src_path}: {e}")


# # ======================
# # 主处理流程
# # ======================

# # 加载训练数据
# df_train = pd.read_csv(f'{rd}/train.csv')
# df_localizers = pd.read_csv(f'{rd}/train_localizers.csv')

# # 调试模式限制数据量
# series_uids = df_train['SeriesInstanceUID'].unique()[:10] if DEBUG else df_train['SeriesInstanceUID'].unique()

# # 构建Series到DICOM路径的映射
# series_dicom_map = {
#     si: glob.glob(os.path.join(rd, 'series', si, '*.dcm'))
#     for si in series_uids
# }

# # 并行排序DICOM文件
# with Pool(cpu_count()) as pool:
#     sorted_results = list(tqdm(pool.imap(sort_series, series_dicom_map.items()),
#                            total=len(series_dicom_map),
#                            desc="Sorting DICOM series"))

# # 生成索引映射表
# rows = []
# for series_uid, sorted_paths in tqdm(sorted_results, desc="生成映射表"):
#     modality = df_train[df_train['SeriesInstanceUID'] == series_uid]['Modality'].iloc[0]
#     for idx, path in enumerate(sorted_paths):
#         sop_uid = os.path.splitext(os.path.basename(path))[0]
#         rows.append({
#             'SeriesInstanceUID': series_uid,
#             'SOPInstanceUID': sop_uid,
#             'dicom_filename': path,
#             'relative_index': idx,
#             'Modality': modality
#         })

# # 保存索引文件
# df_mapping = pd.DataFrame(rows).sort_values(['SeriesInstanceUID', 'relative_index'])
# df_mapping.to_csv('series_index_mapping.csv', index=False)
# print("已保存 series_index_mapping.csv")
df_mapping = pd.read_csv('/kaggle/input/series-index-mapping/series_index_mapping.csv')


# # 处理坐标数据
# df_coords = df_localizers.copy()
# mapping_dict = {
#     (row['SeriesInstanceUID'], row['SOPInstanceUID']): (row['relative_index'], row['dicom_filename'])
#     for _, row in df_mapping.iterrows()
# }

# # 添加相对坐标
# relative_indices, relative_xs, relative_ys = [], [], []
# for _, row in tqdm(df_coords.iterrows(), total=len(df_coords), desc="处理坐标"):
#     key = (row['SeriesInstanceUID'], row['SOPInstanceUID'])
#     if key not in mapping_dict:
#         relative_indices.append(None)
#         relative_xs.append(None)
#         relative_ys.append(None)
#         continue

#     rel_index, dicom_path = mapping_dict[key]
#     relative_indices.append(rel_index)

#     try:
#         ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
#         h, w = int(ds.Rows), int(ds.Columns)
#         coords = eval(row['coordinates']) if isinstance(row['coordinates'], str) else row['coordinates']
        
#         x_rel = (coords['x'] / w) * GLOBAL_WIDTH
#         y_rel = (coords['y'] / h) * GLOBAL_WIDTH
        
#         relative_xs.append(x_rel)
#         relative_ys.append(y_rel)
#     except:
#         relative_xs.append(None)
#         relative_ys.append(None)

# # 保存带相对坐标的文件
# df_coords['relative_index'] = relative_indices
# df_coords['relative_x'] = relative_xs
# df_coords['relative_y'] = relative_ys
# df_coords.to_csv('train_localizers_with_relative.csv', index=False)
# print("已保存 train_localizers_with_relative.csv")
df_coords = pd.read_csv('/kaggle/input/train-localizers-with-relative/train_localizers_with_relative.csv')


# # ======================
# # 图像转换
# # ======================
# # 准备转换任务列表
# outputList = []
# exclude_cols = ['SeriesInstanceUID', 'PatientAge', 'PatientSex', 'Modality', 'Aneurysm Present']
# location_cols = [col for col in df_train.columns if col not in exclude_cols]

# for si in tqdm(series_uids, desc="准备转换任务"):
#     pdf = df_train[df_train['SeriesInstanceUID'] == si]
#     for _, row in pdf.iterrows():
#         locations = [col for col in location_cols if row[col] == 1]
#         if not locations:
#             continue

#         for loc in locations:
#             loc_clean = loc.replace('/', '_')
#             out_dir = f'cvt_png/{loc_clean}/{si}'
            
#             df_series = df_mapping[df_mapping['SeriesInstanceUID'] == si]
#             for _, row_mapping in df_series.iterrows():
#                 outputList.append({
#                     'impath': row_mapping['dicom_filename'],
#                     'dst': f'{out_dir}/{row_mapping["relative_index"]:04d}.png',
#                     'modality': row_mapping['Modality']
#                 })

# # 并行执行转换
# print(f"开始转换DICOM到PNG（共{len(outputList)}个文件）...")
# start_time = time.time()
# Parallel(n_jobs=cpu_count())(
#     delayed(dicom_to_png)(
#         img['impath'], 
#         img['dst'], 
#         GLOBAL_WIDTH, 
#         apply_windowing=True, 
#         modality=img['modality']
#     ) for img in tqdm(outputList)
# )

# # 打印耗时
# elapsed = time.time() - start_time
# hours, rem = divmod(elapsed, 3600)
# minutes, seconds = divmod(rem, 60)
# print(f"转换完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")


# # 安装 7z
# !apt-get install p7zip-full -y

# # 压缩并显示进度
# !7z a -tzip cvt_png.zip /kaggle/working/cvt_png/

