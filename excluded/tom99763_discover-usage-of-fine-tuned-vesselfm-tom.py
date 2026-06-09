import sys
import matplotlib.pyplot as plt 
sys.path.append('/kaggle/input/rsna-iad-vesselfm-codebase')
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
from utils.data import generate_transforms
from utils.io import determine_reader_writer
import os
from monai.transforms import LoadImaged, Spacingd, LoadImage
from monai.networks.nets import DynUNet
import SimpleITK as sitk
import yaml
import torch.nn as nn
from scipy.ndimage import label
from tqdm import  tqdm
import pydicom
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import pandas as pd
from matplotlib.widgets import Slider
import ipywidgets as widgets
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import ast
from tqdm import tqdm


yaml_path = '/kaggle/input/rsna-iad-vesselfm-codebase/configs/inference.yaml'

with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)


class CFG:
    ckpt_path = '/kaggle/input/rsna-iad-vesselfm-finetuned/pytorch/default/1/finetune_rsna_vesselfm-val_volumetric_recall0.7549.ckpt'
    model_structure = {
        'in_channels': 1,
        'out_channels': 1,
        'spatial_dims': 3,
        'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'kernel_size': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        'upsample_kernel_size': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'filters': [32, 64, 128, 256, 320, 320],
        'res_block': True}
    device = 'cuda:0'
    thrd = 0.1

    #sliding window
    batch_size= 1
    patch_size= [128, 128, 128]
    overlap= 0.5
    mode= "constant"
    sigma_scale= 0.125
    padding_mode= "constant"

    #volume transform
    transforms_config = config['transforms_config']
    transforms_config.insert(2, {'Resize': {
        'spatial_size': (None, 512, 512),
        'mode': 'bilinear'
    }},)
    
    tta = config['tta']
    post = config['post']
    merging = config['merging']


def load_model(cfg):
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)['state_dict']
    ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
    model = DynUNet(**CFG.model_structure)
    model.load_state_dict(ckpt)
    model.eval()
    return model.to(cfg.device)


# def load_series2vol(series_path):
#     loader = LoadImage(image_only = True, reader = "ITKReader")
#     volume = loader(series_path)
#     return volume.permute(2, 0, 1)


# def load_series2vol(series_path):
#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(series_path)
#     reader.SetFileNames(dicom_names)
#     image = reader.Execute()
#     volume = sitk.GetArrayFromImage(image)
#     return volume


def load_series2vol(series_path, series_id=None, spacing_tolerance=1e-3, resample=False, default_thickness=1.0):
    reader = sitk.ImageSeriesReader()
    
    # Get all series IDs
    series_ids = reader.GetGDCMSeriesIDs(series_path)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {series_path}")
    
    # Pick first if not specified
    if series_id is None:
        series_id = series_ids[0]
    else:
        series_id = str(series_id)
    
    # Get file names
    all_files = reader.GetGDCMSeriesFileNames(series_path, series_id)
    
    # --- Filter files by consistent size ---
    file_sizes = {}
    for f in all_files:
        img = sitk.ReadImage(f)
        file_sizes.setdefault(img.GetSize(), []).append(f)
    
    # Pick the most common size
    target_size = max(file_sizes, key=lambda k: len(file_sizes[k]))
    files = file_sizes[target_size]
    
    reader.SetFileNames(files)
    image = reader.Execute()
    
    # --- Fix zero thickness ---
    spacing = list(image.GetSpacing())
    if spacing[2] == 0:
        spacing[2] = default_thickness
        image.SetSpacing(spacing)
    
    # --- Optional resample ---
    if resample and abs(spacing[2] - spacing[0]) > spacing_tolerance:
        new_spacing = [spacing[0], spacing[1], spacing[0]]
        new_size = [
            int(round(image.GetSize()[0] * spacing[0] / new_spacing[0])),
            int(round(image.GetSize()[1] * spacing[1] / new_spacing[1])),
            int(round(image.GetSize()[2] * spacing[2] / new_spacing[2]))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)

    volume = sitk.GetArrayFromImage(image)
    
    return volume


def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image

    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def read_dicom_info(path):
    """Read DICOM metadata needed for sorting and decoding."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    try:
        z = float(ds.ImagePositionPatient[2])  # Preferred
    except AttributeError:
        z = float(ds.InstanceNumber)           # Fallback
    return path, z, ds

def read_pixel_data(path, rescale_slope, rescale_intercept):
    """Read pixel data and apply rescale."""
    ds = pydicom.dcmread(path)  # full read with pixels
    arr = ds.pixel_array.astype(np.float32)
    if rescale_slope is not None and rescale_intercept is not None:
        arr = arr * rescale_slope + rescale_intercept
    return arr

def load_volume_fast(directory, max_workers=8):
    # Step 1: List all .dcm files
    dcm_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    if not dcm_files:
        raise RuntimeError(f"No DICOM files found in {directory}")

    # Step 2: Read metadata in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        meta_info = list(executor.map(read_dicom_info, dcm_files))

    # Step 3: Sort by slice position
    meta_info.sort(key=lambda x: x[1])
    sorted_paths = [m[0] for m in meta_info]
    first_ds = meta_info[0][2]

    # Extract rescale params once
    rescale_slope = getattr(first_ds, "RescaleSlope", None)
    rescale_intercept = getattr(first_ds, "RescaleIntercept", None)

    # Step 4: Read one slice to get shape and dimension
    test_arr = pydicom.dcmread(sorted_paths[0]).pixel_array
    if test_arr.ndim == 2:
        depth = len(sorted_paths)
        height, width = test_arr.shape
        volume = np.zeros((depth, height, width), dtype=np.float32)
    elif test_arr.ndim == 3:
        # multi-frame DICOM case
        depth = test_arr.shape[0]
        height, width = test_arr.shape[1], test_arr.shape[2]
        if len(sorted_paths) > 1:
            raise ValueError("Multiple multi-frame DICOM files not supported yet")
        volume = np.zeros((depth, height, width), dtype=np.float32)
    else:
        raise ValueError(f"Unexpected pixel array shape: {test_arr.shape}")

    # Step 5: Load pixel data in parallel directly into volume
    def load_into_array(idx_path):
        idx, path = idx_path
        arr = read_pixel_data(path, rescale_slope, rescale_intercept)
        if arr.ndim == 3 and volume.shape[0] == arr.shape[0]:
            volume[:] = arr  # entire volume from single file
        elif arr.ndim == 2:
            volume[idx] = arr
        else:
            raise ValueError(f"Shape mismatch loading {path}, arr shape: {arr.shape}, volume shape: {volume.shape}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(load_into_array, enumerate(sorted_paths))

    return volume


inferer = SlidingWindowInfererAdapt(
        roi_size=CFG.patch_size, sw_batch_size=CFG.batch_size, overlap=CFG.overlap,
        mode=CFG.mode, sigma_scale=CFG.sigma_scale, padding_mode=CFG.padding_mode
    )

transforms = generate_transforms(CFG.transforms_config)

image_reader_writer = determine_reader_writer('nii')()

model = load_model(CFG)


def inference(path, load_series=False):
    #vol = load_series2vol(path)

    preds = []
    for scale in CFG.tta['scales']:
        if load_series:
            image = load_volume_fast(path)
        else:
            image = image_reader_writer.read_images(path)[0]
        print(image.shape)
        image = transforms(image.astype(np.float32))[None].to(CFG.device)
        #apply test time augmentation
        if CFG.tta['invert']:
            image = 1 - image if image.mean() > CFG.tta['invert_mean_thresh'] else image
            
        if CFG.tta['equalize_hist']:
            image_np = image.cpu().squeeze().numpy()
            image_equal_hist_np = equalize_hist(image_np, nbins=CFG.tta['hist_bins'])
            image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

        original_shape = image.shape
        image = resample(image, factor=scale)
        logits = inferer(image, model)
        logits = resample(logits, target_shape=original_shape)
        preds.append(logits.cpu().squeeze())
    if CFG.merging['max']:
        pred = torch.stack(preds).max(dim=0)[0].sigmoid()
    else:
        pred = torch.stack(preds).mean(dim=0).sigmoid()
    # pred_thresh = (pred > CFG.thrd).numpy()

    # # post-processing
    # if CFG.post['apply']:
    #     pred_thresh = remove_small_objects(
    #         pred_thresh, min_size=CFG.post['small_objects_min_size'],
    #         connectivity=CFG.post['small_objects_connectivity']
    #     )
    return  pred, image


df_loc = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
df_loc['x'] = df_loc['coordinates'].map(lambda x: ast.literal_eval(x)['x'])
df_loc['y'] = df_loc['coordinates'].map(lambda x: ast.literal_eval(x)['y'])


series_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'


series_maps = {}
for series_uid in tqdm(df_loc["SeriesInstanceUID"].unique()):
    series_path_ = f"{series_path}/{series_uid}"
    files = sorted(os.listdir(series_path_))  # ensure consistent order
    # strip .dcm to get SOPInstanceUID
    series_maps[series_uid] = {
        f.replace(".dcm", ""): idx for idx, f in enumerate(files)
    }


# Flatten series_maps into a dataframe
mapping = [
    {"SeriesInstanceUID": series_uid, "SOPInstanceUID": sop_uid, "dcm_idx": int(idx)}
    for series_uid, uid_map in series_maps.items()
    for sop_uid, idx in uid_map.items()
]
df_map = pd.DataFrame(mapping)

# Merge instead of apply
df_loc = df_loc.merge(df_map, on=["SeriesInstanceUID", "SOPInstanceUID"], how="left")


df_loc[df.SeriesInstanceUID == uid]


path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'





df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
df_abnormal = pd.read_csv('/kaggle/input/addition-resource-rsna-iad/multiframe_dicoms.csv')


series_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
paths = sorted(os.listdir(series_path))
path = os.path.join(series_path, paths[0])
len(os.listdir(path))


uid = path.split('/')[-1]

df_sample = df[df.SeriesInstanceUID==uid].copy()


uid in df_abnormal.SeriesInstanceUID.tolist()


df_sample


df.Modality.unique()


CFG.transforms_config


with torch.no_grad():
    p, im = inference(path, True)


depth = p.shape[0]


depth


im.shape


plt.imshow(p[depth//4], alpha = 0.8, cmap='jet')
plt.imshow(im[0, 0, depth//4].cpu(), alpha=0.5, cmap='gray')


def sample_positive_points(segmentation, N, z_scale=1., std_scale=0.5, seed=42, xy_ratio = 2.5, z_ratio=2):
    if seed is not None:
        torch.manual_seed(seed)   # Fix the PyTorch RNG seed
    
    B, _, Z, Y, X = segmentation.shape
    points_list = []

    center_z, center_y, center_x = Z / 2.0, Y / 2.0, X / 2.0
    radius_z = (Z / z_ratio) * z_scale
    radius_xy = min(Y, X) / xy_ratio  # circle radius in XY plane

    device = segmentation.device

    # Create ellipsoid mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, dtype=torch.float32, device=device),
        torch.arange(Y, dtype=torch.float32, device=device),
        torch.arange(X, dtype=torch.float32, device=device),
        indexing='ij'
    )
    ellipsoid_mask = (
        ((zz - center_z) / radius_z) ** 2 +
        ((yy - center_y) ** 2 + (xx - center_x) ** 2) / radius_xy ** 2
    ) <= 1

    for b in range(B):
        seg_masked = segmentation[b, 0] * ellipsoid_mask

        pos_idx = torch.nonzero(seg_masked, as_tuple=False)

        num_candidates = max(N * 10, 1000)

        std_z = radius_z * std_scale
        std_xy = radius_xy * std_scale

        gaussian_samples = torch.empty((num_candidates, 3), device=device).normal_(0, 1)
        gaussian_samples[:, 0] *= std_z
        gaussian_samples[:, 1] *= std_xy
        gaussian_samples[:, 2] *= std_xy
        gaussian_samples += torch.tensor([center_z, center_y, center_x], device=device)

        gaussian_samples = gaussian_samples.round().long()
        gaussian_samples[:, 0] = gaussian_samples[:, 0].clamp(0, Z - 1)
        gaussian_samples[:, 1] = gaussian_samples[:, 1].clamp(0, Y - 1)
        gaussian_samples[:, 2] = gaussian_samples[:, 2].clamp(0, X - 1)

        gaussian_samples = torch.unique(gaussian_samples, dim=0)

        mask_vals = ellipsoid_mask[
            gaussian_samples[:, 0], gaussian_samples[:, 1], gaussian_samples[:, 2]
        ]
        valid_gaussian_points = gaussian_samples[mask_vals]

        seg_vals = segmentation[b, 0][
            valid_gaussian_points[:, 0], valid_gaussian_points[:, 1], valid_gaussian_points[:, 2]
        ]
        valid_pos_gaussian_points = valid_gaussian_points[seg_vals > 0]

        if len(valid_pos_gaussian_points) >= N:
            choice = torch.randperm(len(valid_pos_gaussian_points), device=device)[:N]
            sampled = valid_pos_gaussian_points[choice]
        elif len(pos_idx) >= N:
            choice = torch.randperm(len(pos_idx), device=device)[:N]
            sampled = pos_idx[choice]
        elif len(pos_idx) > 0:
            repeats = (N + len(pos_idx) - 1) // len(pos_idx)
            repeated = pos_idx.repeat((repeats, 1))
            choice = torch.randperm(len(repeated), device=device)[:N]
            sampled = repeated[choice]
        else:
            valid_idx = torch.nonzero(ellipsoid_mask, as_tuple=False)
            choice = torch.randperm(len(valid_idx), device=device)[:N]
            sampled = valid_idx[choice]

        points_list.append(sampled)

    points = torch.stack(points_list, dim=0)  # (B, N, 3)
    return points


mask = p>0.1
mask.sum()


N = 10000 #number of sampling point


points = sample_positive_points(mask[None, None], N)


print(f"sampling ratio: {points.shape[1] * 100/mask.sum():.4f}%")


points.shape[1]


plt.scatter(points[0, :, 1], points[0, :, 2])


from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# Extract coordinates
x, y, z = points[0, :, 0], points[0, :, 1], points[0, :, 2]

# Create 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, alpha=0.5)  # s=1 for small dots

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# %matplotlib inline

# def browse_slices(z):
#     fig, ax = plt.subplots()
#     idx = points[0, :, 0] == z
#     ax.imshow(im[0, 0, z].cpu(), alpha=0.5, cmap='gray')
#     if idx.any():
#         ax.scatter(points[0, idx, 1], points[0, idx, 2], color='r', s=10)
#     ax.set_title(f"Z-slice: {z}")
#     plt.show()

# widgets.interact(browse_slices, z=widgets.IntSlider(min=0, max=im[0,0].shape[0]-1, step=1, value=0))


def create_point_overlay_animation(im, points, interval=200, figsize=(6, 6)):
    """
    im: torch.Tensor (1, 1, Z, H, W) or numpy array (Z, H, W)
    points: numpy array (N, 3) or (B, N, 3) with (z, y, x)
    """

    # Convert tensors to numpy if needed
    if hasattr(im, "cpu"):
        im = im.cpu().numpy()
    if im.ndim == 5:  # (B, C, Z, H, W)
        im = im[0, 0]
    elif im.ndim == 4:  # (C, Z, H, W)
        im = im[0]
    
    if hasattr(points, "cpu"):
        points = points.cpu().numpy()
    if points.ndim == 3:
        points = points[0]

    Z = im.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    img_artist = ax.imshow(im[0], alpha=0.5, cmap='gray')
    scatter_artist = ax.scatter([], [], color='r', s=10)
    ax.set_title(f"Z-slice: 0")

    def update(frame):
        img_artist.set_array(im[frame])
        idx = points[:, 0] == frame
        if np.any(idx):
            scatter_artist.set_offsets(points[idx][:, [2, 1]])  # (x, y)
        else:
            scatter_artist.set_offsets(np.empty((0, 2)))  # ensure 2D empty array
        ax.set_title(f"Z-slice: {frame}")
        return img_artist, scatter_artist
    anim = FuncAnimation(fig, update, frames=Z, interval=interval, blit=True)
    plt.close(fig)
    return anim


anim = create_point_overlay_animation(im, points, interval=200)
HTML(anim.to_jshtml())


class StopForward(Exception): pass

def extract_features_with_hook(model, image, stop_at_layer):
    features = {}
    def hook_fn(module, input, output):
        features['feat'] = output
        raise StopForward()

    target_module = dict(model.named_modules())[stop_at_layer]
    handle = target_module.register_forward_hook(hook_fn)
    try:
        _ = model(image)
    except StopForward:
        pass
    finally:
        handle.remove()
    return features['feat']


def compute_slices(spatial_shape, roi_size, step):
    slices_list = []
    for start_d in range(0, spatial_shape[0], step[0]):
        end_d = start_d + roi_size[0]
        if end_d > spatial_shape[0]:
            start_d = spatial_shape[0] - roi_size[0]
            end_d = spatial_shape[0]

        for start_h in range(0, spatial_shape[1], step[1]):
            end_h = start_h + roi_size[1]
            if end_h > spatial_shape[1]:
                start_h = spatial_shape[1] - roi_size[1]
                end_h = spatial_shape[1]

            for start_w in range(0, spatial_shape[2], step[2]):
                end_w = start_w + roi_size[2]
                if end_w > spatial_shape[2]:
                    start_w = spatial_shape[2] - roi_size[2]
                    end_w = spatial_shape[2]

                slices_list.append((
                    (start_d, end_d),
                    (start_h, end_h),
                    (start_w, end_w)
                ))

    # Remove duplicates by converting to hashable tuples
    slices_list = list(dict.fromkeys(slices_list))

    # Convert tuples back to slices
    slices_list = [
        (slice(d[0], d[1]), slice(h[0], h[1]), slice(w[0], w[1]))
        for d, h, w in slices_list
    ]
    return slices_list


def sliding_window_patches(image, roi_size, device=None, overlap=0.5):
    batch_mode = (image.dim() == 5)
    if batch_mode:
        B = image.shape[0]
        C = image.shape[1]
        spatial_shape = image.shape[2:]
    else:
        B = 1
        C = image.shape[0]
        spatial_shape = image.shape[1:]
        image = image.unsqueeze(0)  # add batch dim

    step = [max(1, int(s * (1 - overlap))) for s in roi_size]
    slices = compute_slices(spatial_shape, roi_size, step)

    for sl in slices:
        patch = image[(slice(None), slice(None)) + sl]  # (B, C, D_roi, H_roi, W_roi)
        coord = (sl[0].start, sl[1].start, sl[2].start)
        if batch_mode:
            yield patch.to(device), coord
        else:
            yield patch[0].to(device), coord


def assign_point_features_with_sliding(model, image, points, stop_at="downsamples.3", roi_size=(128,128,128), overlap=0.5):
    B, N, _ = points.shape
    device = image.device
    point_feats_sum = None
    point_counts = torch.zeros((B, N), device=device)

    for patch, (start_z, start_y, start_x) in sliding_window_patches(image, roi_size, device=device, overlap=overlap):
        with torch.no_grad():
            feat_map = extract_features_with_hook(model, patch, stop_at)
        C_feat = feat_map.shape[1]

        if point_feats_sum is None:
            point_feats_sum = torch.zeros((B, N, C_feat), device=device)

        scale_z = feat_map.shape[2] / patch.shape[2]
        scale_y = feat_map.shape[3] / patch.shape[3]
        scale_x = feat_map.shape[4] / patch.shape[4]

        for b in range(B):
            mask_inside = (
                (points[b, :, 0] >= start_z) & (points[b, :, 0] < start_z + patch.shape[2]) &
                (points[b, :, 1] >= start_y) & (points[b, :, 1] < start_y + patch.shape[3]) &
                (points[b, :, 2] >= start_x) & (points[b, :, 2] < start_x + patch.shape[4])
            )
            inside_idx = torch.nonzero(mask_inside, as_tuple=False).squeeze(1)
            if len(inside_idx) == 0:
                continue

            local_z = ((points[b, inside_idx, 0] - start_z).float() * scale_z).long().clamp(0, feat_map.shape[2]-1)
            local_y = ((points[b, inside_idx, 1] - start_y).float() * scale_y).long().clamp(0, feat_map.shape[3]-1)
            local_x = ((points[b, inside_idx, 2] - start_x).float() * scale_x).long().clamp(0, feat_map.shape[4]-1)

            feats = feat_map[b, :, local_z, local_y, local_x].permute(1,0)  # (num_pts, C_feat)
            point_feats_sum[b, inside_idx] += feats
            point_counts[b, inside_idx] += 1

    point_feats = point_feats_sum / point_counts.clamp_min(1).unsqueeze(-1)
    return point_feats


target_layer = "downsamples.2"


point_feats = assign_point_features_with_sliding(model, im, points, stop_at=target_layer)


point_feats.shape


!pip install /kaggle/input/pip-install-pyg-v2/torch_spline_conv-1.2.2+pt26cu124-cp311-cp311-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg-v2/torch_sparse-0.6.18+pt26cu124-cp311-cp311-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg-v2/pyg_lib-0.4.0+pt26cu124-cp311-cp311-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg-v2/torch_cluster-1.6.3+pt26cu124-cp311-cp311-linux_x86_64.whl
!pip install /kaggle/input/pip-install-pyg-v2/torch_geometric-2.6.1-py3-none-any.whl


from torch_geometric.nn import radius_graph
from torch_cluster import knn_graph
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.data import Data


batch = torch.zeros(points.shape[0], dtype=torch.int64)
edge_index = knn_graph(points[0], k=15, loop=False)
data= Data(points = points[0], x = point_feats[0], edge_index = edge_index, batch = batch)
add_pos = AddRandomWalkPE(walk_length=8, attr_name=None)
data = add_pos(data)


data.x.shape #feature_dim + walk_length




