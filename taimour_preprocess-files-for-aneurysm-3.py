# =========================
# Imports & Global Settings
# =========================
import os
import gc
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import shutil
from functools import partial


# Competition constants
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Paths - Kaggle-specific
TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
OUTPUT_DIR = "/kaggle/working/preprocessed_data"

# Processing configuration
TARGET_SIZE = (64, 64, 64)      # final (D,H,W)
TARGET_SPACING_MM = 1.0         # isotropic resample
CTA_WINDOW = (300.0, 700.0)     # (center, width) for CT (CTA)
MRI_Z_CLIP = 3.0                # clip z-score to Â±3Ïƒ
MAX_SERIES = None               # Set to None for full dataset, or number for testing
TEST_MODE = False               # Ultra-fast test mode (5 series, no progress bars)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Preprocessing output will be saved to: {OUTPUT_DIR}")


# ==========================
# DICOM Processing Utilities
# ==========================
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)  # avoid zeros/negatives
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) using safe zoom."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)


# ==========================
# DICOM Series Processor
# ==========================
class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes."""
    
    def __init__(
        self,
        target_size: Tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: Tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
    ):
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip
        
        # Adjustment counters
        self.slope_adjustments = 0
        self.intercept_adjustments = 0
        self.adaptive_windowing_count = 0
        self.failed_series = []
        self.modalities = {}  # Track modality for each series

    def _validate_and_apply_rescale(self, sl: np.ndarray, ds) -> np.ndarray:
        """Validate slope/intercept values and apply rescaling."""
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        
        # Validate slope
        if slope <= 0 or not np.isfinite(slope) or abs(slope) > 1000:
            slope = 1.0
            self.slope_adjustments += 1
        
        # Validate intercept with range clamping for extreme values
        if not np.isfinite(intercept):
            intercept = 0.0
            self.intercept_adjustments += 1
        elif abs(intercept) > 10000:
            intercept = np.clip(intercept, -2000, 0)
            self.intercept_adjustments += 1
        
        # Apply rescaling directly to array (no copy)
        sl *= slope
        sl += intercept
        
        # Validate result
        if np.any(~np.isfinite(sl)):
            sl = np.nan_to_num(sl, copy=False)
        
        # Post-rescale range check
        min_val = sl.min()
        max_val = sl.max()
        if min_val < -5000 or max_val > 10000:
            np.clip(sl, -3000, 5000, out=sl)
        
        return sl

    def _log_adjustment_summary(self):
        """Log summary of adjustments made during processing."""
        print(f"\nProcessing adjustments summary:")
        print(f"- Slope adjustments: {self.slope_adjustments}")
        print(f"- Intercept adjustments: {self.intercept_adjustments}")
        print(f"- Adaptive windowing: {self.adaptive_windowing_count}")
        print(f"- Failed series: {len(self.failed_series)}")
        if self.failed_series:
            print("\nFirst 5 failed series:")
            for sid in self.failed_series[:5]:
                print(f"  {sid}")

    def _process_single_series(self, series_path: str, series_id: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Return (D,H,W) float32 volume in [0,1]. Returns None if processing fails."""
        try:
            # Collect DICOM datasets
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.endswith(".dcm"):
                        try:
                            ds = pydicom.dcmread(os.path.join(root, f), force=True)
                            if hasattr(ds, "PixelData"):
                                dicoms.append(ds)
                        except Exception as e:
                            continue
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")

            dicoms = self._sort_slices(dicoms)
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

            # Choose base HxW
            base_h, base_w = self._choose_base_shape(dicoms)

            modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()
            self.modalities[series_id] = modality_tag  # Track modality
            
            vol_slices = []
            for ds in dicoms:
                arr = ds.pixel_array
                # standardize to (N,H,W) where N=number of frames (1 if 2D)
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape(n, h, w)
                    frames = arr
                else:
                    frames = arr[np.newaxis, ...]  # shape (1,H,W)

                for sl in frames:
                    sl = sl.astype(np.float32)

                    # Handle MONOCHROME1 inversion
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl

                    # Apply validated rescaling
                    sl = self._validate_and_apply_rescale(sl, ds)

                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if len(vol_slices) == 0:
                raise ValueError("No valid slices extracted.")

            volume = np.stack(vol_slices, axis=0).astype(np.float32)  # (D,H,W)

            # Normalize by modality -> [0,1]
            volume = self._normalize_by_modality(volume, modality_tag)

            # Isotropic resample (mm-based)
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                newD = max(1, int(round(z * dz / self.target_spacing_mm)))
                newH = max(1, int(round(y * dy / self.target_spacing_mm)))
                newW = max(1, int(round(x * dx / self.target_spacing_mm)))
                volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Resize to target grid
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)
            
            # Quality check - reject near-zero volumes
            if np.mean(volume) < 0.01 or np.std(volume) < 0.01:
                raise ValueError("Volume has near-zero intensity")
                
            return volume, modality_tag

        except Exception as e:
            self.failed_series.append(series_id)
            return None, None
    
    def process_series(self, series_path: str, series_id: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Wrapper for parallel processing - returns (series_id, volume)"""
        volume, modality = self._process_single_series(series_path, series_id)
        return (series_id, volume, modality)
    
    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]; col = orient[3:]
            normal = np.cross(row, col)
            def sort_key(ds):
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))
            return sorted(ds_list, key=sort_key)
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
        try:
            dy, dx = map(float, ds_sorted[0].PixelSpacing)
        except Exception:
            ps = getattr(ds_sorted[0], "PixelSpacing", [1.0, 1.0])
            dy, dx = float(ps[0]), float(ps[1])

        if has_multiframe:
            dz = float(getattr(ds_sorted[0], "SpacingBetweenSlices", getattr(ds_sorted[0], "SliceThickness", 1.0)))
        else:
            zs = []
            for i in range(1, len(ds_sorted)):
                p0 = np.array(getattr(ds_sorted[i-1], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                p1 = np.array(getattr(ds_sorted[i], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                d = np.linalg.norm(p1 - p0)
                if d > 0:
                    zs.append(d)
            if zs:
                dz = float(np.median(zs))
            else:
                dz = float(getattr(ds_sorted[0], "SliceThickness", 1.0))

        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        shapes = []
        for ds in ds_list:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
            except Exception:
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
            
            # Check if values are in normal CT range
            if min_val >= -2000 and max_val <= 4000:
                # Normal range: use standard windowing
                c, w = self.cta_window
                lo, hi = c - w / 2.0, c + w / 2.0
            else:
                # Extreme range: use adaptive windowing
                self.adaptive_windowing_count += 1
                
                # Percentile-based adaptive window
                p1, p99 = np.percentile(volume, [1, 99])
                margin = (p99 - p1) * 0.1
                lo = p1 - margin
                hi = p99 + margin
                
                # Ensure minimum window width
                if hi - lo < 100:
                    center = (hi + lo) / 2
                    lo = center - 50
                    hi = center + 50
            
            # In-place clipping and normalization
            np.clip(volume, lo, hi, out=volume)
            volume = (volume - lo) / (hi - lo + 1e-6)
            return volume.astype(np.float32, copy=False)
        else:
            # MRI processing
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            
            # Validate statistics
            if std < 1e-6 or not np.isfinite(mean) or not np.isfinite(std):
                return np.full_like(volume, 0.5, dtype=np.float32)
            
            # Check dynamic range
            min_val, max_val = volume.min(), volume.max()
            if max_val - min_val < 1e-6:
                return np.full_like(volume, 0.5, dtype=np.float32)
            
            # In-place operations
            volume -= mean
            volume /= std
            zc = float(self.mri_z_clip)
            np.clip(volume, -zc, zc, out=volume)
            volume = (volume + zc) / (2.0 * zc)
            return volume.astype(np.float32, copy=False)



# ==============================
# Preprocessing Execution
# ==============================

def process_single_series_wrapper(args):
    """Wrapper for parallel processing to handle the processor instance correctly."""
    processor, series_path, series_id = args
    return processor.process_series(series_path, series_id)
    
def main():
    # Initialize processor
    processor = DICOMProcessor(
        target_size=TARGET_SIZE,
        target_spacing_mm=TARGET_SPACING_MM,
        cta_window=CTA_WINDOW,
        mri_z_clip=MRI_Z_CLIP,
    )
    
    # Load training labels
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Loaded {len(train_df)} series from train.csv")
    
    # CORRECTED SAMPLING LOGIC - Replace the existing block with this
    if MAX_SERIES is not None and len(train_df) > MAX_SERIES:
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=MAX_SERIES, random_state=42)
        _, test_idx = next(sss.split(train_df, train_df['Aneurysm Present']))  # Get TEST indices
        train_df = train_df.iloc[test_idx].reset_index(drop=True)
        print(f"Reduced to {len(train_df)} series for testing (stratified)")

    # TEST_MODE for ultra-fast validation (5 series)
    if TEST_MODE:
        print("\nâš ï¸� TEST MODE ACTIVE - Processing only 5 series for validation")
        train_df = train_df.head(5).copy()
        MAX_WORKERS = 2  # Fewer workers for small test
    else:
        # Determine optimal number of workers (leave 1 core free)
        MAX_WORKERS = max(1, os.cpu_count() - 1)
        
    # Clear output directory for the selected subset to force reprocessing
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Cleared output directory to force reprocessing of {len(train_df)} series")
    
    # Prepare processing tasks
    tasks = []
    for _, row in train_df.iterrows():
        series_id = row[ID_COL]
        series_path = os.path.join(SERIES_DIR, series_id)
        tasks.append((processor, series_path, series_id))
    
    # Parallel processing
    print(f"\nStarting preprocessing of {len(train_df)} series using {MAX_WORKERS} workers...")
    start_time = time.time()
    processed_volumes = {}
    series_modalities = {}  # Will store modality for each series
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_series = {
            executor.submit(process_single_series_wrapper, task): task[2] 
            for task in tasks
        }
        
        # Progress tracking
        completed = 0
        pbar = tqdm(total=len(tasks), disable=TEST_MODE)  # Hide progress in test mode
        
        for future in as_completed(future_to_series):
            result = future.result()
            series_id = result[0]
            volume = result[1]
            modality = result[2] if len(result) > 2 else None
            
            if volume is not None:
                output_path = os.path.join(OUTPUT_DIR, f"{series_id}.npy")
                np.save(output_path, volume)
                processed_volumes[series_id] = volume
                series_modalities[series_id] = modality  # Store modality
            
            completed += 1
            pbar.update(1)
            pbar.set_postfix(processed=completed, failed=len(tasks)-completed)
            
            # Memory cleanup
            if completed % 10 == 0:
                gc.collect()
    
    pbar.close()
    total_time = time.time() - start_time
    print(f"\nPreprocessing completed in {total_time:.1f} seconds")
    print(f"Average time per series: {total_time/max(len(train_df),1):.2f} seconds")
    
    # Save metadata
    metadata = {
        'series_ids': list(processed_volumes.keys()),
        'modalities': series_modalities,
        'config': {
            'target_size': TARGET_SIZE,
            'target_spacing_mm': TARGET_SPACING_MM,
            'cta_window': CTA_WINDOW,
            'mri_z_clip': MRI_Z_CLIP,
        },
        'processing_time': total_time,
        'series_count': len(processed_volumes),
        'failed_series_count': len(train_df) - len(processed_volumes)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final report
    processor._log_adjustment_summary()

    # Additional modality breakdown
    if series_modalities:
        modality_counts = {}
        for m in series_modalities.values():
            if m:
                modality_counts[m] = modality_counts.get(m, 0) + 1
        
        print("\nModality distribution in processed data:")
        for mod, count in modality_counts.items():
            print(f"- {mod}: {count} series ({count/len(series_modalities)*100:.1f}%)")
            
    print(f"\nPreprocessing complete! Successfully processed {len(processed_volumes)}/{len(train_df)} series")
    print(f"Preprocessed volumes saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {os.path.join(OUTPUT_DIR, 'metadata.json')}")

if __name__ == "__main__":
    main()

