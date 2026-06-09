!rm /kaggle/working/* -rf


!pip install --no-index --no-deps --find-links=/kaggle/input/rsna-submission-v1-wheels-data/wheelhouse hydra-core==1.3.2 monai==1.5.1



import gc
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
import pydicom
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pydicom.dataset import FileDataset
from pydicom.errors import InvalidDicomError
from scipy import ndimage
import torch

LOGGER = logging.getLogger("submission_v1")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

np.random.seed(42)
torch.manual_seed(42)
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using device: %s", device)




# --- User configuration: update these paths for your Kaggle notebook run --- #
REPO_ROOT = Path("/kaggle/input/kaggle-rsna2025/Kaggle-RSNA2025")  # Update if the repository lives elsewhere
if not REPO_ROOT.exists():
    candidate_roots = [
        Path("/kaggle/input/kaggle-rsna2025"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidate_roots:
        candidate = candidate.resolve()
        if (candidate / "configs/submission_model/v1.yaml").exists():
            REPO_ROOT = candidate
            break

CONFIG_PATH = REPO_ROOT / "configs/submission_model/v1.yaml"
CHECKPOINT_PATH = Path("/kaggle/input/kaggle-rsna2025/Kaggle-RSNA2025/weights/best_model.pth")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}. Update REPO_ROOT.")
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError("Checkpoint not found. Update CHECKPOINT_PATH to your trained weights.")

sys.path.append(str(REPO_ROOT))
LOGGER.info("Repository root: %s", REPO_ROOT)




from utils.builders import build_cls_model, build_data_aug

cfg = OmegaConf.load(CONFIG_PATH)
label_cols = list(cfg.training.dataset.label_cols)
ID_COL = cfg.training.dataset.id_col
LOGGER.info("Loaded config with %d label columns", len(label_cols))

val_aug_mapping = build_data_aug(cfg.validation.data_aug)
val_aug = val_aug_mapping.get("augmentation")
model = build_cls_model(cfg.model.backbone, cfg.model.cls_head)

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
if isinstance(state, dict) and any(k.startswith("state_dict") for k in state.keys()):
    state = state.get("state_dict", state.get("model_state_dict"))
model.load_state_dict(state, strict=True)
model = model.to(device)
model.eval()
LOGGER.info("Model parameters loaded from %s", CHECKPOINT_PATH)




# Processing / Model config
TARGET_SIZE = (96, 96, 96)  # final (D,H,W)
TARGET_SPACING_MM = 1.0  # isotropic resample
CTA_WINDOW = (300.0, 700.0)  # (center, width) for CT (CTA)
MRI_Z_CLIP = 3.0  # clip z-score to +/- 3 sigma
SLOPE_MAX_ABS = 1000.0
INTERCEPT_MAX_ABS = 10000.0
RESCALE_MIN_LIMIT = -5000.0
RESCALE_MAX_LIMIT = 10000.0
CT_MIN_NORMAL = -2000.0
CT_MAX_NORMAL = 4000.0
MIN_WINDOW_WIDTH = 100.0
SMALL_STD_EPS = 1e-6
MULTIFRAME_DIM_THRESHOLD = 3


def _safe_zoom(volume: np.ndarray, zoom_factors: tuple[float, ...], order: int = 1) -> np.ndarray:
    """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(SMALL_STD_EPS, f)) for f in zoom_factors)  # avoid zeros/negatives
    if len(zf) != volume.ndim:
        zf = zf[: volume.ndim] if len(zf) > volume.ndim else (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)


def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) using safe zoom."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)


class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes."""

    def __init__(
        self,
        target_size: tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
    ) -> None:
        """Initialize resampling, windowing, and normalization parameters."""
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip

        # Adjustment counters
        self.slope_adjustments = 0
        self.intercept_adjustments = 0
        self.adaptive_windowing_count = 0

    def _validate_and_apply_rescale(self, sl: np.ndarray, ds: FileDataset) -> np.ndarray:
        """Validate slope/intercept values and apply rescaling."""
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))

        # Validate slope
        if slope <= 0 or not np.isfinite(slope) or abs(slope) > SLOPE_MAX_ABS:
            slope = 1.0
            self.slope_adjustments += 1

        # Validate intercept with range clamping for extreme values
        if not np.isfinite(intercept):
            intercept = 0.0
            self.intercept_adjustments += 1
        elif abs(intercept) > INTERCEPT_MAX_ABS:
            intercept = np.clip(intercept, CT_MIN_NORMAL, 0)
            self.intercept_adjustments += 1

        # Apply rescaling
        rescaled = sl * slope + intercept

        # Validate result
        if np.any(~np.isfinite(rescaled)):
            rescaled = np.nan_to_num(rescaled, copy=False)

        # Post-rescale range check
        min_val, max_val = rescaled.min(), rescaled.max()
        if min_val < RESCALE_MIN_LIMIT or max_val > RESCALE_MAX_LIMIT:
            rescaled = np.clip(rescaled, RESCALE_MIN_LIMIT * 0.6, RESCALE_MAX_LIMIT * 0.5)

        return rescaled

    def log_adjustment_summary(self) -> None:
        """Log summary of adjustments made during processing."""
        LOGGER.info(
            "Processing adjustments - Slope: %s, Intercept: %s, Adaptive windowing: %s",
            self.slope_adjustments,
            self.intercept_adjustments,
            self.adaptive_windowing_count,
        )

    def load_dicom_series(self, series_path: Path | str) -> np.ndarray:
        """Return (D,H,W) float32 volume in [0,1]."""
        try:
            return self._load_dicom_series(Path(series_path))
        except (OSError, RuntimeError, ValueError):
            LOGGER.warning("Failed to load series %s", series_path, exc_info=True)
            return np.zeros(self.target_size, dtype=np.float32)

    def _load_dicom_series(self, path_obj: Path) -> np.ndarray:
        dicoms = self._collect_dicoms(path_obj)
        dicoms = self._sort_slices(dicoms)
        modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()
        has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
        spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

        base_h, base_w = self._choose_base_shape(dicoms)
        vol_slices = self._build_slices(dicoms, base_h, base_w)
        if not vol_slices:
            message = "No valid slices extracted."
            raise ValueError(message) from None

        volume = np.stack(vol_slices, axis=0).astype(np.float32)
        volume = self._normalize_by_modality(volume, modality_tag)

        if self.target_spacing_mm is not None:
            dz, dy, dx = spacing
            z, y, x = volume.shape
            new_d = max(1, round(z * dz / self.target_spacing_mm))
            new_h = max(1, round(y * dy / self.target_spacing_mm))
            new_w = max(1, round(x * dx / self.target_spacing_mm))
            volume = _safe_zoom(volume, (new_d / z, new_h / y, new_w / x), order=1)

        tz, ty, tx = self.target_size
        z, y, x = volume.shape
        return _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

    def _collect_dicoms(self, path_obj: Path) -> list[FileDataset]:
        dicoms: list[FileDataset] = []
        for dicom_path in path_obj.rglob("*.dcm"):
            try:
                ds = pydicom.dcmread(dicom_path, force=True)
            except (InvalidDicomError, OSError, ValueError):
                LOGGER.debug("Skipping unreadable DICOM %s", dicom_path, exc_info=LOGGER.isEnabledFor(logging.DEBUG))
                continue
            if hasattr(ds, "PixelData"):
                dicoms.append(ds)
        if not dicoms:
            message = f"No valid DICOM files with pixel data in {path_obj}"
            raise ValueError(message) from None
        return dicoms

    def _build_slices(
        self,
        dicoms: list[FileDataset],
        base_h: int,
        base_w: int,
    ) -> list[np.ndarray]:
        slices: list[np.ndarray] = []
        for ds in dicoms:
            arr = ds.pixel_array
            if arr.ndim >= MULTIFRAME_DIM_THRESHOLD:
                h, w = arr.shape[-2], arr.shape[-1]
                frame_count = int(np.prod(arr.shape[:-2]))
                frames = arr.reshape(frame_count, h, w)
            else:
                frames = arr[np.newaxis, ...]

            for frame in frames:
                slice_data = frame.astype(np.float32)
                if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                    slice_data = slice_data.max() - slice_data

                slice_data = self._validate_and_apply_rescale(slice_data, ds)
                slices.append(_resize_slice(slice_data, base_h, base_w))
        return slices

    def _sort_slices(self, ds_list: list[pydicom.dataset.FileDataset]) -> list[pydicom.dataset.FileDataset]:
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]
            col = orient[3:]
            normal = np.cross(row, col)

            def sort_key(ds: FileDataset) -> float:
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))

            return sorted(ds_list, key=sort_key)
        except (AttributeError, TypeError) as exc:
            debug_context = LOGGER.isEnabledFor(logging.DEBUG)
            LOGGER.debug("Falling back to instance number sort due to %s", exc, exc_info=debug_context)
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(
        self, ds_sorted: list[pydicom.dataset.FileDataset], *, has_multiframe: bool = False
    ) -> tuple[float, float, float]:
        try:
            dy, dx = map(float, ds_sorted[0].PixelSpacing)
        except (AttributeError, TypeError, ValueError) as exc:
            debug_context = LOGGER.isEnabledFor(logging.DEBUG)
            LOGGER.debug("Using default pixel spacing for %s due to %s", ds_sorted[0], exc, exc_info=debug_context)
            ps = getattr(ds_sorted[0], "PixelSpacing", [1.0, 1.0])
            dy, dx = float(ps[0]), float(ps[1])

        if has_multiframe:
            dz = float(
                getattr(
                    ds_sorted[0],
                    "SpacingBetweenSlices",
                    getattr(ds_sorted[0], "SliceThickness", 1.0),
                )
            )
        else:
            zs = []
            for i in range(1, len(ds_sorted)):
                p0 = np.array(
                    getattr(ds_sorted[i - 1], "ImagePositionPatient", [0, 0, 0]),
                    dtype=np.float32,
                )
                p1 = np.array(
                    getattr(ds_sorted[i], "ImagePositionPatient", [0, 0, 0]),
                    dtype=np.float32,
                )
                d = np.linalg.norm(p1 - p0)
                if d > 0:
                    zs.append(d)
            dz = float(np.median(zs)) if zs else float(getattr(ds_sorted[0], "SliceThickness", 1.0))

        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: list[pydicom.dataset.FileDataset]) -> tuple[int, int]:
        shapes = []
        for ds in ds_list:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
            except (AttributeError, TypeError, ValueError):
                arr = ds.pixel_array
                h, w = arr.shape[-2], arr.shape[-1]
            shapes.append((h, w))
        vals, counts = np.unique(shapes, return_counts=True, axis=0)
        base = tuple(vals[counts.argmax()])
        return int(base[0]), int(base[1])

    def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
        """CT: adaptive windowing for extreme ranges; MR: z-score -> clip -> [0,1]."""
        volume = np.nan_to_num(volume, copy=False)

        if modality_tag == "CT":
            min_val, max_val = volume.min(), volume.max()

            if min_val >= CT_MIN_NORMAL and max_val <= CT_MAX_NORMAL:
                c, w = self.cta_window
                lo, hi = c - w / 2.0, c + w / 2.0
            else:
                self.adaptive_windowing_count += 1

                p1, p99 = np.percentile(volume, [1, 99])
                margin = (p99 - p1) * 0.1
                lo = p1 - margin
                hi = p99 + margin

                if hi - lo < MIN_WINDOW_WIDTH:
                    center = (hi + lo) / 2
                    half_width = MIN_WINDOW_WIDTH / 2
                    lo = center - half_width
                    hi = center + half_width

            v = np.clip(volume, lo, hi)
            v = (v - lo) / (hi - lo + 1e-6)
            return v.astype(np.float32, copy=False)
        # MRI processing
        mean = float(volume.mean())
        std = float(volume.std() + 1e-6)

        # Validate statistics
        if std < SMALL_STD_EPS or not np.isfinite(mean) or not np.isfinite(std):
            return np.full_like(volume, 0.5, dtype=np.float32)

        # Check dynamic range
        min_val, max_val = volume.min(), volume.max()
        if max_val - min_val < SMALL_STD_EPS:
            return np.full_like(volume, 0.5, dtype=np.float32)

        v = (volume - mean) / std
        zc = float(self.mri_z_clip)
        v = np.clip(v, -zc, zc)
        v = (v + zc) / (2.0 * zc)
        return v.astype(np.float32, copy=False)


processor = DICOMProcessor(
    target_size=TARGET_SIZE,
    target_spacing_mm=TARGET_SPACING_MM,
    cta_window=CTA_WINDOW,
    mri_z_clip=MRI_Z_CLIP,
)
LOGGER.info("DICOM processor ready")




def _prepare_tensor(volume: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(volume).unsqueeze(0)  # (C, D, H, W)
    tensor = val_aug(tensor)
    return tensor.unsqueeze(0).to(device, non_blocking=True)  # (1, C, D, H, W)


@torch.no_grad()
def predict(series_path: str) -> pl.DataFrame:
    series_id = Path(series_path).name
    LOGGER.info(f"predict: {series_id}")
    volume = processor.load_dicom_series(series_path)
    inputs = _prepare_tensor(volume)
    logits = model(inputs)
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    row = {ID_COL: series_id}
    row.update({label: float(prob) for label, prob in zip(label_cols, probs, strict=True)})
    gc.collect()
    return pl.DataFrame(row)


LOGGER.info("Predict function is ready")




import kaggle_evaluation.rsna_inference_server

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))


