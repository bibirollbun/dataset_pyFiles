deps_path = '/kaggle/input/czii-cryoet-dependencies'


! cp -r /kaggle/input/czii-cryoet-dependencies/asciitree-0.3.3/ asciitree-0.3.3/


! pip wheel asciitree-0.3.3/asciitree-0.3.3/



!pip install asciitree-0.3.3-py3-none-any.whl


! pip install -q --no-index --find-links {deps_path} --requirement {deps_path}/requirements.txt


!pip install /kaggle/input/tensorrt-10-1-0/nvidia_cuda_runtime_cu12-12.2.140-py3-none-manylinux1_x86_64.whl
!pip install /kaggle/input/tensorrt-10-1-0/tensorrt_cu12_bindings-10.1.0-cp310-none-manylinux_2_17_x86_64.whl
!pip install /kaggle/input/tensorrt-10-1-0/tensorrt_cu12_libs-10.1.0-py2.py3-none-manylinux_2_17_x86_64.whl
!pip install /kaggle/input/tensorrt-10-1-0/tensorrt_cu12-10.1.0-py2.py3-none-any.whl
!pip install /kaggle/input/tensorrt-10-1-0/tensorrt-10.1.0-py2.py3-none-any.whl
!pip install /kaggle/input/tensorrt-10-1-0/polygraphy-0.49.14-py2.py3-none-any.whl


from typing import List, Tuple, Union
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)


import sys
sys.path.append('/kaggle/input/czii-ckpt/')
from UNetWithMem import UNetWithMem
sys.path.append('/kaggle/input/czii-weight-by-fbeta')
from UNet3d import ResUNet3D
from DLinkNet3D import DLinkNet3D
sys.path.append('/kaggle/input/czii-radius-0-5')
from UNetWithMemRes import UNetWithMemRes


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    
    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension
        
    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]
        
    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)
    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
    
    return positions

def extract_3d_patches_minimal_overlap(arrays: List[np.ndarray], patch_size: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    
    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_size : int
        Size of cubic patches (a x a x a)
        
    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")
    
    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    
    if patch_size > min(shape):
        raise ValueError(f"patch_size ({patch_size}) must be smaller than smallest dimension {min(shape)}")
    
    m, n, l = shape
    patches = []
    coordinates = []
    
    # Calculate starting positions for each dimension
    x_starts = calculate_patch_starts(m, patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)
    
    # Extract patches from each array
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x:x + patch_size,
                        y:y + patch_size,
                        z:z + patch_size
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))
    
    return patches, coordinates

# Note: I should probably averge the overlapping areas, 
# but here they are just overwritten by the most recent one. 

def reconstruct_array(patches: List[np.ndarray], 
                     coordinates: List[Tuple[int, int, int]], 
                     original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Reconstruct array from patches.
    
    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int]
        Shape of the original array
        
    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape, dtype=np.int64)  # To track overlapping regions
    
    patch_size = patches[0].shape[0]
    
    for patch, (x, y, z) in zip(patches, coordinates):
        reconstructed[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ] = patch
        
    
    return reconstructed


import pandas as pd

def dict_to_df(coord_dict, experiment_name):
    """
    Convert dictionary of coordinates to pandas DataFrame.
    
    Parameters:
    -----------
    coord_dict : dict
        Dictionary where keys are labels and values are Nx3 coordinate arrays
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'label']
    """
    # Create lists to store data
    all_coords = []
    all_labels = []
    
    # Process each label and its coordinates
    for label, coords in coord_dict.items():
        all_coords.append(coords)
        all_labels.extend([label] * len(coords))
    
    # Concatenate all coordinates
    all_coords = np.vstack(all_coords)
    
    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x': all_coords[:, 0],
        'y': all_coords[:, 1],
        'z': all_coords[:, 2]
    })

    
    return df


TRAIN_DATA_DIR = "/kaggle/input/create-numpy-dataset"
TEST_DATA_DIR = "/kaggle/input/czii-cryo-et-object-identification"


cp -r /kaggle/input/tensorrt-10-1-0/torch2trt-master /kaggle/working/torch2trt


!pip install /kaggle/working/torch2trt


from torch2trt import TRTModule
import tensorrt as trt
from cuda import cudart

LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(LOGGER, "")

class Net:
    def __init__(self, weights, device=0):
        self.device = device
        
        torch.cuda.set_device(self.device)
        
        self.runtime = trt.Runtime(LOGGER)
        with open(weights, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.trt_model = TRTModule(
            input_names=['images'],
            output_names=['output'],
            engine=self.engine
        )
    
    def __call__(self, img):
        if img.device.index != self.device:
            img = img.cuda(self.device)
        
        with torch.no_grad():
            output = self.trt_model(img)
        return output

        
class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models  = torch.nn.ModuleList(models)
    
    def forward(self, x):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = [model(x) for model in self.models]
        return sum(outputs) / len(outputs)



import torch
import tensorrt as trt
from cuda import cudart
import torch.nn as nn

class TRTEnsembleModel(nn.Module):
    def __init__(self, models, weights):
        super(TRTEnsembleModel, self).__init__()
        self.models = []
        self.weights = weights
        for model_config in models:
            self.models.append(Net(weights=model_config['engine_path'], device=model_config['device']))
    
    def forward(self, x):
        with torch.no_grad():
            outputs = [model(x) * weight for model, weight in zip(self.models, self.weights)]
        return sum(outputs)

class TRTEnsembleModel_diff_channel(nn.Module):
    def __init__(self, models_7_channel, models_8_channel, weights1, weights2, new_e_model, new_e_model_weight):
        super(TRTEnsembleModel_diff_channel, self).__init__()
        self.models_7_channel = []
        self.models_8_channel = []
        self.selected_channels = [0, 1, 3, 4, 5, 6]
        self.weights1 = weights1
        self.weights2 = weights2
        self.new_e_model_weight = new_e_model_weight
        
        # 7-channel models
        for model_config in models_7_channel:
            self.models_7_channel.append(Net(weights=model_config['engine_path'], device=model_config['device']))
            
        # 8-channel models
        for model_config in models_8_channel:
            self.models_8_channel.append(Net(weights=model_config['engine_path'], device=model_config['device']))
            
        # New ensemble model
        self.new_e_model = Net(weights=new_e_model['engine_path'], device=new_e_model['device'])
    
    def forward(self, x):
        with torch.no_grad():
            outputs1 = [model(x)[:, self.selected_channels, :, :, :] * weight 
                       for model, weight in zip(self.models_7_channel, self.weights1)]
            outputs2 = [model(x)[:, self.selected_channels, :, :, :] * weight 
                       for model, weight in zip(self.models_8_channel, self.weights2)]
            outputs3 = self.new_e_model(x) * self.new_e_model_weight

        all_outputs = outputs1 + outputs2
        return sum(all_outputs) + outputs3

def load_trt_model(weight1, weight2, new_e_model, new_e_model_weight):
    """
    weight1, weight2: リストで、各要素は辞書で以下を含む:
    {
        'engine_path': str,  # .engineファイルのパス
        'device': int        # GPUデバイス番号
    }
    """
    final_model = TRTEnsembleModel_diff_channel(
        models_7_channel=weight1,
        models_8_channel=weight2,
        weights1=[0.05+0.0125],  # 元の実装の重みを維持
        weights2=[0.05+0.0125, 0.05+0.0125, 0.05+0.0125, 0.1+0.0125],  # 元の実装の重みを維持
        new_e_model=new_e_model,
        new_e_model_weight=new_e_model_weight
    )
    return final_model

def load_trt_model_same_c(weight):
    """
    weight: リストで、各要素は辞書で以下を含む:
    {
        'engine_path': str,  # .engineファイルのパス
        'device': int        # GPUデバイス番号
    }
    """
    ens_weights = [0.05+0.0125, 0.1+0.0125, 0.05+0.0125]  # 元の実装の重みを維持
    final_model = TRTEnsembleModel(weight, ens_weights)
    return final_model


# 同じチャンネル数のモデルのアンサンブル
weight = [
    # {'engine_path': '/path/to/model1.engine', 'device': 0},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model1/UNet3D.engine', 'device': 0},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model1/UNetWithMem.engine', 'device': 0},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model1/DLinkNet3D.engine', 'device': 0}
]

ensemble_model1 = load_trt_model_same_c(weight)

# 異なるチャンネル数のモデルのアンサンブル
weight1 = [
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/UNet_0.engine', 'device': 1}
]

weight2 = [
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/UNet_1.engine', 'device': 1},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/UNetWithMem_2.engine', 'device': 1},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/UNet_3.engine', 'device': 1},
    {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/UNetWithMemRes_4.engine', 'device': 1}
]

new_e_model = {'engine_path': '/kaggle/input/make-trt-main-ensemble-original/ensemble_model2/load_3_models_ensemble.engine', 'device': 1}

ensemble_model2 = load_trt_model(weight1, weight2, new_e_model, 0.4)


ensemble_models = [ensemble_model1,ensemble_model2]


import json
copick_config_path = TRAIN_DATA_DIR + "/copick.config"

with open(copick_config_path) as f:
    copick_config = json.load(f)

copick_config['static_root'] = '/kaggle/input/czii-cryo-et-object-identification/test/static'

copick_test_config_path = 'copick_test.config'

with open(copick_test_config_path, 'w') as outfile:
    json.dump(copick_config, outfile)
import copick

root = copick.from_file(copick_test_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
tomo_type = "denoised"

inference_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS")
])
import cc3d

id_to_name = {1: "apo-ferritin", 
              2: "beta-galactosidase", 
              3: "ribosome", 
              4: "thyroglobulin", 
              5: "virus-like-particle"}


from monai.inferers import sliding_window_inference
import time
import concurrent.futures

BLOB_THRESHOLD = {1: 2, 2: 33, 3: 78, 4: 42, 5: 400}
CERTAINTY_THRESHOLD = 0.8
    
classes = [1, 2, 3, 4, 5]

is_sw = True

def run_inference(ensemble, device, input_tensor, size, sw_bs):
    pred = 0
    tta_ct = 0
    FLIP_TTA = True
    FLIP_TTA_LIST = [[2]]
    weights = [1.0]
    ROTATE_TTA = False
    ROTATE_K_LIST = [1,2,3]
    OVERLAP_T = 0.15
    out_device = 'cpu'
    
    with torch.cuda.device(device):
        with torch.amp.autocast(device_type='cuda'):  # Corrected device_type
            pred += sliding_window_inference(
                input_tensor,
                roi_size=[size, size, size],
                sw_batch_size=sw_bs,
                sw_device=device,
                device=out_device,  # Keep on GPU
                predictor=ensemble,
                overlap=OVERLAP_T,
                progress=False
            )
            tta_ct += 1
            if FLIP_TTA:
                for dims in FLIP_TTA_LIST:
                    t_pred = sliding_window_inference(
                        inputs=torch.flip(input_tensor, dims=dims),
                        roi_size=[size, size, size],
                        sw_batch_size=sw_bs,
                        predictor=ensemble,
                        overlap=OVERLAP_T,
                        device=out_device,  # Keep on GPU
                        sw_device=device,
                        progress=False
                    )
                    pred += torch.flip(t_pred, dims=dims)
                    tta_ct += 1
            if ROTATE_TTA:
                for k in ROTATE_K_LIST:
                    t_pred = sliding_window_inference(
                        inputs=torch.rot90(input_tensor, k=k, dims=[3,4]),
                        roi_size=[size, size, size],
                        sw_batch_size=sw_bs,
                        predictor=ensemble,
                        overlap=OVERLAP_T,
                        device=out_device,  # Keep on GPU
                        sw_device=device,
                        progress=False
                    )
                    pred += torch.rot90(t_pred, k=-k, dims=[3,4])
                    tta_ct += 1
    pred /= tta_ct
    return pred.cpu() # Move to CPU once after processing

def run_inference_tta(ensemble, device, input_tensor, size, sw_bs):
    pred = 0
    tta_ct = 0
    FLIP_TTA_LIST = [[2],[3]]
    with torch.cuda.device(device):
        with torch.amp.autocast(device_type='cuda'):
            for dims in FLIP_TTA_LIST:
                t_pred = sliding_window_inference(
                    inputs=torch.flip(input_tensor, dims=dims),
                    roi_size=[size, size, size],
                    sw_batch_size=sw_bs,
                    predictor=ensemble,
                    overlap=0.15,
                    device=device,  # Keep on GPU
                    sw_device=device,
                    progress=False
                )
                pred += torch.flip(t_pred, dims=dims)
                tta_ct += 1
    pred /= tta_ct
    return pred.cpu()  # Move to CPU once after processing

# Start the timer
import time
start_time = time.time()

device_ids = ["cuda:0", "cuda:1"]
sizes = [128, 180]
sw_bs = [2, 2]
inference_fns = [run_inference, run_inference]  # run_inference_tta

start_time = time.time()

with torch.no_grad():
    location_df = []
    for run in root.runs:
        print(run)
    
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()

        tomo_patched_data = [{"image": tomo}]
        tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0, progress=False)
        st = time.time()
        
        input_tensor = tomo_ds[0]["image"].unsqueeze(0).pin_memory()  # Pinned memory
        
        inputs = [input_tensor.to(device, non_blocking=True) for device in device_ids]  # Non-blocking transfer
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(inference_fns[i], ensemble_models[i], device_ids[i], inputs[i], sizes[i], sw_bs[i])
                for i in range(2)
            ]
            outputs = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        averaged_output = sum(outputs)  # Outputs are already on CPU
        en = time.time()
        print(f'use time: {en-st}s')
        
        reconstructed_mask = torch.softmax(averaged_output, dim=1)[0]
        _, reconstructed_mask = (reconstructed_mask > CERTAINTY_THRESHOLD).max(0)
        reconstructed_mask = reconstructed_mask.cpu().numpy()
                

        
        location = {}
        for c in classes:
            cc = cc3d.connected_components(reconstructed_mask == c)
            stats = cc3d.statistics(cc)
            zyx = stats['centroids'][1:] * 10.012444
            zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD[c]]
            xyz = np.ascontiguousarray(zyx_large[:, ::-1])    
            location[id_to_name[c]] = xyz
    
        df = dict_to_df(location, run.name)
        location_df.append(df)
        
    location_df = pd.concat(location_df)




# End the timer
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time

# Calculate the processing time for 500 voxels
time_per_3_voxels = elapsed_time
time_per_500_voxels = (elapsed_time / 3) * 500

# Convert the time for 500 voxels from seconds to hours
time_per_500_voxels_hours = time_per_500_voxels / 3600

# Print the results
print(f"The processing time for 3 voxels is {elapsed_time} seconds")
print(f"The processing time for 500 voxels is approximately {time_per_500_voxels_hours} hours")


location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))


import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# 假设sub已经给定，拼接DataFrame
# df = pd.concat([sub[0], sub[1], sub[2]], ignore_index=True)
# 粒子半径映射
# 'beta-amylase'
particle_names = ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
particle_radius = {
    'apo-ferritin': 60,
    # 'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}
df = location_df.copy()

final = []  # 用于存储最终的结果
for pidx, p in enumerate(particle_names):
    # 筛选出该粒子类型的所有点
    pdf = df[df['particle_type'] == p].reset_index(drop=True)
    p_rad = particle_radius[p]
    
    # 根据 experiment 分组
    grouped = pdf.groupby(['experiment'])
    
    for exp, group in grouped:
        group = group.reset_index(drop=True)
        
        # 使用DBSCAN进行聚类
        coords = group[['x', 'y', 'z']].values
        db = DBSCAN(eps=p_rad, min_samples=2, metric='euclidean').fit(coords)
        labels = db.labels_
        
        # 将聚类结果添加到DataFrame中
        group['cluster'] = labels
        
        # 对每个簇进行处理
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # 跳过噪声点
            
            cluster_points = group[group['cluster'] == cluster_id]
            
            # 计算簇的中心（平均位置）
            avg_x = cluster_points['x'].mean()
            avg_y = cluster_points['y'].mean()
            avg_z = cluster_points['z'].mean()
            
            # 更新簇内点的位置
            group.loc[group['cluster'] == cluster_id, ['x', 'y', 'z']] = avg_x, avg_y, avg_z
            group = group.drop_duplicates(subset=['x', 'y', 'z'])
        # 将处理后的数据添加到 final 列表
        final.append(group)

# 合并处理后的数据
df_save = pd.concat(final, ignore_index=True)
df_save = df_save.drop(columns=['cluster'])

# 排序按 'experiment' 和 'particle_type' 两列
df_save = df_save.sort_values(by=['experiment', 'particle_type']).reset_index(drop=True)

# 重新生成 'id' 列，从 1 开始
df_save['id'] = np.arange(0, len(df_save))

# 输出结果到 CSV
df_save.to_csv('submission.csv', index=False)




