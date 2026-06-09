!uv pip install monai dynamic_network_architectures cucim-cu12 fire


!mkdir inference


%%writefile inference/_gpu_resampling.py

from copy import deepcopy
from typing import Union, Tuple, List

import numpy as np
import cupy as cp
from cucim.skimage.transform import resize as cucim_resize
from cupyx.scipy.ndimage import map_coordinates as cucim_map_coordinates

ANISO_THRESHOLD = 3


def get_do_separate_z(
    spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    anisotropy_threshold=ANISO_THRESHOLD,
):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
    return axis


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array(
        [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    )
    return new_shape


def determine_do_sep_z_and_axis(
    force_separate_z: bool,
    current_spacing,
    new_spacing,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
) -> Tuple[bool, Union[int, None]]:
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]
    return do_separate_z, axis


def resample_data_or_seg_to_spacing(
    data: np.ndarray,
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold
    )

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(
        data, new_shape, axis, order, do_separate_z, order_z=order_z
    )
    return data_reshaped


def resample_data_or_seg_to_shape(
    data: np.ndarray,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
):
    """
    needed for segmentation export. Stupid, I know
    """
    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold
    )

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(
        data, new_shape, axis, order, do_separate_z, order_z=order_z
    )
    return data_reshaped


def resample_data_or_seg(
    data: np.ndarray,
    _new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    axis: Union[None, int] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out=None,
):
    """
    cuCIM/cupy-accelerated version of resample_data_or_seg
    separate_z=True will resample with order 0 along z
    :param data: numpy array (c, x, y, z)
    :param new_shape:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return: numpy array
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(_new_shape) == data.ndim - 1

    # Convert to GPU
    data_gpu = cp.asarray(data, dtype=cp.float32)
    kwargs = {"mode": "edge", "anti_aliasing": False}
    shape = cp.array(data_gpu[0].shape)
    new_shape = cp.array(_new_shape)

    if dtype_out is None:
        dtype_out = cp.float32

    reshaped_final = cp.zeros(
        tuple([data_gpu.shape[0]] + new_shape.tolist()), dtype=dtype_out
    )

    if cp.any(shape != new_shape):
        data_gpu = data_gpu.astype(cp.float32, copy=False)

        if do_separate_z:
            assert (
                axis is not None
            ), "If do_separate_z, we need to know what axis is anisotropic"

            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data_gpu.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = cp.zeros(tuple(tmp.tolist()))

                # GPU-accelerated slice processing with cuCIM
                for slice_id in range(int(shape[axis])):
                    if axis == 0:
                        reshaped_here[slice_id] = cucim_resize(
                            data_gpu[c, slice_id], new_shape_2d, order, **kwargs
                        )
                    elif axis == 1:
                        reshaped_here[:, slice_id] = cucim_resize(
                            data_gpu[c, :, slice_id], new_shape_2d, order, **kwargs
                        )
                    else:
                        reshaped_here[:, :, slice_id] = cucim_resize(
                            data_gpu[c, :, :, slice_id], new_shape_2d, order, **kwargs
                        )

                if shape[axis] != new_shape[axis]:
                    # GPU-accelerated coordinate mapping with cupyx
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    # align_corners=False - same logic as original
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = cp.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = cp.array([map_rows, map_cols, map_dims])

                    # GPU-accelerated coordinate mapping
                    reshaped_final[c] = cucim_map_coordinates(
                        reshaped_here, coord_map, order=order_z, mode="nearest"
                    )[None]
                else:
                    reshaped_final[c] = reshaped_here
        else:
            # GPU-accelerated direct resize
            for c in range(data_gpu.shape[0]):
                reshaped_final[c] = cucim_resize(
                    data_gpu[c], new_shape, order, **kwargs
                )

        # Convert back to CPU numpy array
        return cp.asnumpy(reshaped_final)
    else:
        # No resampling needed
        return data


%%writefile inference/_gpu_seg_resampling.py

from typing import Union, Tuple, List
import numpy as np
import cupy as cp
import torch
from cucim.skimage.transform import resize as cucim_resize


def compute_new_shape(
    old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
) -> np.ndarray:
    """Compute new shape after resampling"""
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array(
        [int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)]
    )
    return new_shape


def resample_segmentation_to_spacing(
    data: np.ndarray,
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
):
    """
    Resample segmentation to new spacing using nearest neighbor

    Args:
        data: segmentation data with shape (c, x, y, z)
        current_spacing: current spacing (x, y, z)
        new_spacing: target spacing (x, y, z)

    Returns:
        resampled segmentation with same channel dimension
    """
    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    return resample_segmentation_to_shape(data, new_shape)


def resize_segmentation_cupy(segmentation, new_shape, order=3):
    """
    CuPy version of resize_segmentation
    Input: CuPy array
    Output: CuPy array
    """
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(
        new_shape
    ), "new shape must have same dimensionality as segmentation"

    if order == 0:
        return cucim_resize(
            segmentation.astype(cp.float32),
            new_shape,
            order,
            mode="edge",
            clip=True,
            anti_aliasing=False,
        ).astype(tpe)
    else:
        reshaped = cp.zeros(new_shape, dtype=segmentation.dtype)

        unique_labels = cp.sort(cp.unique(segmentation.ravel()))
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = cucim_resize(
                mask.astype(cp.float32),
                new_shape,
                order,
                mode="edge",
                clip=True,
                anti_aliasing=False,
            )
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped


def resample_segmentation_to_shape(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
):
    """
    Resample segmentation to new shape using nearest neighbor

    Args:
        data: segmentation data with shape (c, x, y, z)
        new_shape: target shape (x, y, z)

    Returns:
        resampled segmentation
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    return _resample_segmentation_core_cupy(data, new_shape)


def _resample_segmentation_core_cupy(
    data: np.ndarray,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    dtype_out=None,
):
    """
    Core segmentation resampling function - CuPy accelerated
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    shape = np.array(data[0].shape)
    new_shape_np = np.array(new_shape)

    if dtype_out is None:
        dtype_out = data.dtype

    # Early return if no resampling needed
    if np.all(shape == new_shape_np):
        return data

    # Convert to GPU
    data_gpu = cp.asarray(data, dtype=cp.float32)

    # Allocate output on GPU
    reshaped_final_gpu = cp.zeros((data_gpu.shape[0], *new_shape), dtype=cp.float32)

    # Direct resize with nearest neighbor for all channels
    for c in range(data_gpu.shape[0]):
        reshaped_final_gpu[c] = resize_segmentation_cupy(
            data_gpu[c], new_shape, order=0  # Always nearest neighbor for segmentation
        )

    return cp.asnumpy(reshaped_final_gpu).astype(dtype_out)



%%writefile inference/_transform.py

import numpy as np
import torch
from typing import Union, Optional
from monai.transforms import Transform
from monai.utils import convert_to_tensor
from skimage.morphology import remove_small_objects


class RemoveSmallObjectsDense(Transform):
    """
    Remove small connected components from dense label image.
    Works directly on labeled arrays without one-hot conversion.

    Args:
        min_size: Minimum component size in voxels to keep.
        connectivity: Neighborhood connectivity (1, 2, or 3 for 3D images).
        device: 'cpu' or 'cuda'. Auto-detects from input if None.
    """

    def __init__(
        self, min_size: int = 64, connectivity: int = 1, device: Optional[str] = None
    ):
        super().__init__()
        self.min_size = min_size
        self.connectivity = connectivity
        self.device = device

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            img: Dense label image of shape (H, W, D) with integer labels.

        Returns:
            Image with small objects removed.
        """
        device = self.device or (
            "cuda" if isinstance(img, torch.Tensor) and img.is_cuda else "cpu"
        )

        if device == "cuda":
            return self._process_cpu(
                img.to(device="cpu") if isinstance(img, torch.Tensor) else img
            )
        return self._process_cpu(img)

    def _process_cpu(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """CPU implementation using scikit-image."""
        img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img

        # scikit-image's remove_small_objects works directly on labeled images
        cleaned = img_np * remove_small_objects(
            img_np.astype(bool), min_size=self.min_size, connectivity=self.connectivity
        )

        return convert_to_tensor(cleaned, dtype=torch.uint8)

    def _process_gpu(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("GPU version not implemented yet.")


%%writefile inference/inferer.py

import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, cast
from pathlib import Path
from monai.transforms import NormalizeIntensity
from monai.inferers.utils import sliding_window_inference
from dynamic_network_architectures.architectures.unet import (
    ResidualEncoderUNet,
    UNetDecoder,
)
import nibabel as nib
import monai.transforms as mt
from inference._gpu_resampling import (
    resample_data_or_seg,
    compute_new_shape,
    determine_do_sep_z_and_axis,
)
from inference._transform import RemoveSmallObjectsDense

from inference._gpu_seg_resampling import resample_segmentation_to_spacing


def add_additional_channel_to_state_dict(model_weight, model):
    state_dict = model_weight["network_weights"]

    # Get current model's state dict to identify which keys need modification
    model_state = model.state_dict()

    # Find all layers with channel dimension mismatch
    for key in list(state_dict.keys()):
        if key in model_state:
            pretrained_shape = state_dict[key].shape
            current_shape = model_state[key].shape

            # Check if this is a weight tensor with channel dimension mismatch
            if len(pretrained_shape) == 5 and pretrained_shape != current_shape:
                # This is likely a conv weight: [out_channels, in_channels, D, H, W]
                if pretrained_shape[1] == 1 and current_shape[1] == 2:
                    print(f"Expanding {key} from {pretrained_shape} to {current_shape}")

                    # Create new tensor with correct shape
                    new_weight = torch.zeros(current_shape)

                    # Copy pretrained weights to channel 0
                    new_weight[:, 0:1, :, :, :] = state_dict[key]

                    # Initialize channel 1 with N(0, 0.01)
                    torch.nn.init.normal_(
                        new_weight[:, 1:2, :, :, :], mean=0.0, std=0.01
                    )

                    # Replace in state dict
                    state_dict[key] = new_weight

    return state_dict


class ResEncoderUNetModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        pretrained_weights_path: str | None = None,
        keep_decoder_weights: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.model = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=5,
            num_classes=num_classes,
            features_per_stage=[32, 64, 128, 256, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=(3, 3, 3, 3, 3),
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)),  # type: ignore
            n_blocks_per_stage=(1, 3, 4, 6, 6),
            n_conv_per_stage_decoder=(1, 1, 1, 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"affine": True, "eps": 1e-3},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0.20},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )

        if pretrained_weights_path is not None:
            model_weight = torch.load(pretrained_weights_path, weights_only=False)
            for key in list(model_weight["network_weights"].keys()):
                if key.startswith("decoder.seg_layer"):
                    del model_weight["network_weights"][key]

            state_dict = add_additional_channel_to_state_dict(model_weight, self.model)

            self.model.load_state_dict(state_dict, strict=False)

            if not keep_decoder_weights:
                print("Reinitializing decoder weights...")
                self.model.decoder = UNetDecoder(
                    encoder=self.model.encoder,
                    num_classes=num_classes,
                    n_conv_per_stage=(1, 1, 1, 1),
                    deep_supervision=deep_supervision,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:

        output = self.model(x)
        if self.deep_supervision:
            if self.training:
                return output[::-1]
            else:
                return output[0]

        return output


class Inferer:
    def __init__(
        self,
        model_save_path: str,
        device="cuda",
        inference_roi_sizes: list[tuple[int, int, int]] = [(40, 160, 160)],
        inference_batch_size: int = 8,
        inference_overlap=0.25,
        inference_dtype=torch.float32,
        use_amp: bool = False,
    ):
        self.model_save_path = model_save_path
        self.device = torch.device(device)
        self.inference_roi_sizes = inference_roi_sizes
        self.inference_batch_size = inference_batch_size
        self.inference_overlap = inference_overlap
        self.inference_dtype = inference_dtype
        self.use_amp = use_amp

        if inference_dtype == torch.float32:
            print(
                "[WARNING] Cannot use mixed precision with float32, setting use_amp to False."
            )
            self.use_amp = False

        self._post_transforms = mt.Compose(
            [
                mt.AsDiscrete(argmax=True),
                mt.EnsureType(dtype=torch.uint8),
                RemoveSmallObjectsDense(min_size=500, connectivity=5),
            ],
        )
        self._model = self._load_model()

    def _load_model(self):

        model = ResEncoderUNetModel(
            input_channels=1,
            num_classes=15,
            deep_supervision=False,
            pretrained_weights_path=None,
        )

        weights = torch.load(self.model_save_path, map_location=self.device)
        model.load_state_dict(weights["network_weights"])
        model.to(self.device)
        model.eval()

        return model

    def infer(self, volume: torch.Tensor) -> torch.Tensor:

        volume = volume.unsqueeze(0).to(device=self.device)  # (1, C, H, W, D)
        output_sum = None
        num_scales = len(self.inference_roi_sizes)
    
        for inference_size in self.inference_roi_sizes:
            if self.use_amp:
                with torch.autocast(
                    device_type=self.device.type, dtype=self.inference_dtype
                ):
                    with torch.inference_mode():
                        output = sliding_window_inference(
                            inputs=volume,
                            roi_size=inference_size,
                            sw_batch_size=self.inference_batch_size,
                            predictor=lambda x: self._model(x)[:, :-1, ...], # remove auxiliary head
                            overlap=self.inference_overlap,
                            mode="gaussian",
                            sw_device=self.device,
                            # device=self.device,
                            device="cpu",
                        )
    
            else:
                with torch.inference_mode():
                    output = sliding_window_inference(
                        inputs=volume,
                        roi_size=inference_size,
                        sw_batch_size=self.inference_batch_size,
                        predictor=self._model,
                        overlap=self.inference_overlap,
                        mode="gaussian",
                        sw_device=self.device,
                        # device=self.device,
                        device="cpu"
                    )
    
            assert isinstance(output, torch.Tensor)
            output = output.squeeze(0)
            
            # Accumulate in-place instead of appending to list
            if output_sum is None:
                output_sum = output.clone()  # First iteration: create accumulator
            else:
                output_sum.add_(output)  # Subsequent iterations: in-place addition
            
        # Compute average in-place
        output = output_sum.div_(num_scales)
        output = self._post_transforms(output)
    
        return output.detach().cpu()


def _reorient_to_lps(volume, info_dict={}):
    """Reorient volume and segmentation to LPS coordinate system."""
    orientation = nib.orientations.io_orientation(volume.affine)
    lps_orientation = nib.orientations.axcodes2ornt(("L", "P", "S"))
    transform = nib.orientations.ornt_transform(orientation, lps_orientation)

    info_dict["original_orientation"] = nib.orientations.ornt2axcodes(orientation)
    info_dict["target_orientation"] = "LPS"

    volume_reoriented = nib.orientations.apply_orientation(
        volume.get_fdata(), transform
    )

    volume_affine = np.dot(
        volume.affine, nib.orientations.inv_ornt_aff(transform, volume.shape)
    )

    return nib.Nifti1Image(volume_reoriented, volume_affine)


def _resample_volume_nnunet(volume, spacing=(1.0, 1.0, 1.0), info_dict={}):
    """
    CHANGE: Replaced simple scipy zoom with nnUNet's anisotropic-aware resampling
    """
    current_spacing = volume.header.get_zooms()[:3]
    # Get data and add channel dimension for nnUNet format (c, x, y, z)
    volume_data = volume.get_fdata()[np.newaxis, ...]  # Add channel dim

    # Compute new shape using nnUNet utilities
    new_shape = compute_new_shape(volume_data.shape[1:], current_spacing, spacing)

    # Determine if separate z-axis handling is needed
    do_separate_z, axis = determine_do_sep_z_and_axis(None, current_spacing, spacing)

    volume_resampled = resample_data_or_seg(
        volume_data,
        new_shape,
        axis=axis,
        order=3,
        do_separate_z=do_separate_z,
        order_z=0,
    )

    # Remove channel dimension
    volume_resampled = volume_resampled[0]

    # FIXED: Proper affine matrix handling
    zoom_factors = np.array(current_spacing) / np.array(spacing)

    volume_affine = volume.affine.copy()

    # Scale the existing affine matrix to account for resampling
    # This preserves rotation, shear, and orientation information
    volume_affine[:3, :3] = volume_affine[:3, :3] / zoom_factors
    volume_img = nib.Nifti1Image(volume_resampled, volume_affine)

    return volume_img


def _get_preprocessed_volume(
    volume_path: str,
) -> nib.Nifti1Image:
    nii_volume = nib.load(volume_path)
    nii_npy = nii_volume.get_fdata()
    if len(nii_npy.shape) == 4:
        # Take first channel if multiple channels exist
        nii_npy = nii_npy[..., 0]
        nii_volume = nib.Nifti1Image(nii_npy, nii_volume.affine, nii_volume.header)
    nii_volume = _reorient_to_lps(nii_volume)
    nii_volume = _resample_volume_nnunet(nii_volume, spacing=(0.4425, 0.4425, 0.80))
    return nii_volume


def generate_segmentation(
    volume: np.ndarray,
    inferer: Inferer,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:

    volume_data = torch.from_numpy(volume)
    volume_data = volume_data.unsqueeze(0)
    volume_data = cast(torch.Tensor, NormalizeIntensity()(volume_data))
    volume_data = volume_data.permute(0, 3, 1, 2)  # (C, D, H, W)
    start_time = time.time()
    segmentation = inferer.infer(volume_data)
    print(f"Finished generation in {time.time() - start_time:.2f} seconds.")
    segmentation = segmentation.numpy()
    segmentation = np.transpose(segmentation, (0, 2, 3, 1))

    segmentation = resample_segmentation_to_spacing(
        segmentation,
        current_spacing=current_spacing,
        new_spacing=target_spacing,
    )
    segmentation = segmentation.astype(np.uint8)
    return segmentation


def generate_and_write_segmentation(
    volume_path: str,
    inferer: Inferer,
    output_path: str,
    target_spacing: tuple[float, float, float],
):
    nii_volume = _get_preprocessed_volume(volume_path)
    assert nii_volume.affine is not None

    current_spacing = nii_volume.header.get_zooms()[:3]
    assert len(current_spacing) == 3
    segmentation = generate_segmentation(
        nii_volume.get_fdata(),
        inferer,
        current_spacing=current_spacing,
        target_spacing=target_spacing,
    )
    segmentation = segmentation[0]

    zoom_factors = np.array(current_spacing) / np.array(target_spacing)
    seg_affine = nii_volume.affine.copy()
    seg_affine[:3, :3] = seg_affine[:3, :3] / zoom_factors
    seg_img = nib.Nifti1Image(segmentation, seg_affine)
    nib.save(seg_img, output_path)


import json
from pathlib import Path

_MR_Z_THRESHOLD = 1.95  # All below this will use 3d method
series_path = list(Path("/kaggle/input/rsna-niftii-raw/mra_niftii/mra_processed").iterdir())

valid_series = []
for series in series_path:
    with open(series / "metadata.json", "r") as f:
        metadata = json.load(f)

    spacing = metadata["spacing"][:3]
    if spacing[-1] <= _MR_Z_THRESHOLD:
        valid_series.append(str(series))




def main(volume_paths: str, gpu_id: str, output_path: str):
    import torch
    import time
    import gc
    from inference.inferer import Inferer, generate_and_write_segmentation
    import cupy as cp

    cp.cuda.Device(gpu_id).use()
    
    print(f"Received {len(volume_paths)} volumes for inference on GPU id: {gpu_id}.")

        
    _MR_SPACING_3D = (0.43, 0.43, 0.60)  # MRA 3D
    _MR_Z_THRESHOLD = 1.95  # All below this will use 3d method


    inferer = Inferer(
        model_save_path="/kaggle/input/rsna-niftii-raw/greedy_soup.pth",
        inference_dtype=torch.bfloat16,
        inference_roi_sizes=[(40, 144, 144), (48, 160, 160)],
        inference_overlap=(0.75, 0.625, 0.625),
        inference_batch_size=21,
        use_amp=True,
        device=f"cuda:{gpu_id}",
    )

    for i, path in enumerate(volume_paths):
        volume_path = Path(path) / "volume.nii"
        seg_path = Path(output_path) / volume_path.parent.name
        seg_path.mkdir(parents=True, exist_ok=True)
        seg_path = seg_path / "segmentation.nii.gz"
        
        start_time = time.time()
        generate_and_write_segmentation(
            volume_path=str(volume_path),
            inferer=inferer,
            output_path=str(seg_path),
            target_spacing=_MR_SPACING_3D,
        )

        print(f"Finished {volume_path.parent.name} in {time.time() - start_time} seconds.")
        torch.cuda.empty_cache()
        gc.collect()



import multiprocessing as mp
import torch

NUM_GPUS = 4
TOTAL_SERIES = len(valid_series)

def split_list_n_parts(lst, n):
    """Split list into n equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

print(f"Splitting {TOTAL_SERIES} series between {NUM_GPUS} gpus")

chunks = split_list_n_parts(valid_series, NUM_GPUS)
GPU_IDS = ["0", "1", "2", "3"]

print("total chunks: ", len(chunks))


process_list = []

for chunk, gpu_id in zip(chunks, GPU_IDS):
    p = mp.Process(target=main, args=(chunk, gpu_id, "./mr_segmentation"))
    p.start()
    process_list.append(p)

for p in process_list:
    p.join()




