# === Cell 0: Imports, config, seeding, util print ===
import os, sys, glob, random, textwrap, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import SimpleITK as sitk
from scipy import ndimage as ndi

# Matplotlib aesthetics
mpl.rcParams["figure.dpi"] = 140
mpl.rcParams["font.size"] = 11

def seed_all(s=1337):
    random.seed(s); np.random.seed(s)
seed_all(1337)

print("Python:", sys.version)
print("SimpleITK:", sitk.Version_VersionString())

# CT windowing helpers
def window_image(img_np, center=400, width=1800):
    lo = center - width/2.0
    hi = center + width/2.0
    x = np.clip(img_np, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return x.astype(np.float32)

# quick border extractor for label images
def boundary_from_label(lbl2d):
    # binary edge for each class != 0 (union), using morphological erosion
    mask = (lbl2d > 0).astype(np.uint8)
    er = ndi.binary_erosion(mask, structure=np.ones((3,3)), iterations=1)
    edge = (mask ^ er).astype(np.uint8)
    return edge

# multi-color overlay of label map
def overlay_segmentation(img2d_0to1, lbl2d, alpha=0.35):
    h,w = img2d_0to1.shape
    base = np.stack([img2d_0to1]*3, axis=-1)  # gray->RGB
    uniq = sorted([u for u in np.unique(lbl2d) if u>0])
    # stable color palette across runs
    rng = np.random.default_rng(42)
    colors = rng.random((max(uniq+[1])+1, 3))  # indexed by label id
    # emphasize C1..C7 labels (1..7) with a fixed palette
    fixed = {
        1:(0.85,0.10,0.10), 2:(0.10,0.55,0.85), 3:(0.10,0.75,0.30),
        4:(0.75,0.55,0.10), 5:(0.60,0.10,0.75), 6:(0.15,0.80,0.80), 7:(0.90,0.30,0.30)
    }
    for k,v in fixed.items(): colors[k]=v
    out = base.copy()
    for u in uniq:
        mask = (lbl2d==u)
        color = colors[u]
        out[mask] = (1-alpha)*out[mask] + alpha*np.array(color)
    # thin boundary lines for crispness
    edge = boundary_from_label(lbl2d)
    out[edge.astype(bool)] = (0,0,0)  # black edge
    return out



# === Cell 0P: Patch overlay_segmentation to handle arbitrary label IDs safely ===
def overlay_segmentation(img2d_0to1, lbl2d, alpha=0.35):
    # img2d_0to1: (H,W) in [0,1], lbl2d: (H,W) integer labels
    h, w = img2d_0to1.shape
    base = np.stack([img2d_0to1]*3, axis=-1)
    lbl2d = lbl2d.astype(np.int32)

    uniq = [int(u) for u in np.unique(lbl2d) if u > 0]
    if not uniq:
        return base  # no labels to overlay

    # Ensure palette big enough for cervical (1..7) and thoracic (8..19)
    max_label = max(max(uniq), 19)   # allocate up to 19
    rng = np.random.default_rng(42)
    colors = rng.random((max_label + 1, 3))

    # Fixed readable colors for C1..C7
    fixed = {
        1:(0.85,0.10,0.10), 2:(0.10,0.55,0.85), 3:(0.10,0.75,0.30),
        4:(0.75,0.55,0.10), 5:(0.60,0.10,0.75), 6:(0.15,0.80,0.80),
        7:(0.90,0.30,0.30)
    }
    for k, v in fixed.items():
        if k <= max_label:
            colors[k] = v

    out = base.copy()
    for u in uniq:
        mask = (lbl2d == u)
        out[mask] = (1 - alpha) * out[mask] + alpha * colors[u]

    # crisp boundary
    edge = boundary_from_label(lbl2d)
    out[edge.astype(bool)] = (0, 0, 0)
    return out



# === Cell 1: Discover RSNA 2022 CT dataset with vertebra segmentations ===

CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection-2",  # safety
    "/kaggle/input/rsna-2022-spine-fracture-detection",             # occasional alias
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if Path(r).exists():
        DATA_ROOT = Path(r); break

assert DATA_ROOT is not None, "RSNA 2022 dataset not found under /kaggle/input/**. Add it in 'Add data'."

TRAIN_IMG_DIR = DATA_ROOT/"train_images"
SEG_DIR       = DATA_ROOT/"segmentations"  # official folder name in competition data
CSV_TRAIN     = DATA_ROOT/"train.csv"

print(f"[CHECK] DATA_ROOT: {DATA_ROOT}")
print(f"[CHECK] TRAIN_IMG_DIR exists: {TRAIN_IMG_DIR.exists()}")
print(f"[CHECK] SEG_DIR exists: {SEG_DIR.exists()}")
print(f"[CHECK] CSV_TRAIN exists: {CSV_TRAIN.exists()}")

# Enumerate studies (folders named by StudyInstanceUID)
study_dirs = sorted([p for p in TRAIN_IMG_DIR.iterdir() if p.is_dir()])
print(f"[EDA] #train studies found: {len(study_dirs)} (showing first 5)")
for p in study_dirs[:5]: print("   ", p.name)

# Enumerate segmentation files (NIfTI .nii or .nii.gz)
seg_files = sorted(list(SEG_DIR.glob("*.nii*"))) if SEG_DIR.exists() else []
print(f"[EDA] #segmentations found: {len(seg_files)} (showing first 5)")
for f in seg_files[:5]: print("   ", f.name)

# Build mapping study_id -> segmentation path (by filename match)
seg_map = {}
for f in seg_files:
    stem = f.name.split(".nii")[0]
    seg_map[stem] = f
coverage = sum(1 for p in study_dirs if p.name in seg_map)
print(f"[EDA] Segmentation coverage: {coverage}/{len(study_dirs)} studies with masks")

# Load the label CSV (fracture labels; useful for cross-checks, not needed for stenosis)
if CSV_TRAIN.exists():
    df_train = pd.read_csv(CSV_TRAIN)
    print("[EDA] train.csv shape:", df_train.shape)
    display(df_train.head(3))



# === Cell 2: DICOM series loader, NIfTI loader, and geometry alignment ===

def pick_axial_series(study_folder: Path):
    """
    Heuristic: prefer series path containing 'AX'/'AXIAL' and with the most DICOM files.
    Falls back to the subfolder with most files.
    """
    cand = [p for p in study_folder.rglob("*") if p.is_dir()]
    dcm_dirs = []
    for c in cand:
        try:
            n = sum(1 for _ in c.glob("*.dcm"))
            if n >= 8:  # skip tiny series
                hint = int(("AX" in c.name.upper()) or ("AXIAL" in c.name.upper()))
                dcm_dirs.append((hint, n, c))
        except Exception:
            pass
    if not dcm_dirs:
        return None
    dcm_dirs.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return dcm_dirs[0][2]

def load_dicom_volume(series_dir: Path):
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not uids: 
        return None, None
    # choose UID with max slices
    best_uid, best_files = None, []
    for uid in uids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), uid)
        if len(files) > len(best_files):
            best_uid, best_files = uid, files
    reader.SetFileNames(best_files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    return img, arr

def load_seg_nifti(seg_path: Path):
    seg_img = sitk.ReadImage(str(seg_path))  # supports .nii.gz
    seg_arr = sitk.GetArrayFromImage(seg_img)  # (Z,Y,X)
    return seg_img, seg_arr

def resample_like(moving_img, reference_img, is_label=False):
    """
    Resample 'moving_img' into 'reference_img' geometry (spacing, origin, direction).
    Nearest neighbor for labels, linear for images.
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resamp = sitk.ResampleImageFilter()
    resamp.SetReferenceImage(reference_img)
    resamp.SetInterpolator(interp)
    resamp.SetTransform(sitk.Transform())
    resamp.SetDefaultPixelValue(0)
    out = resamp.Execute(moving_img)
    return out

def mid_sagittal_index(vol_arr):
    # pick mid X column
    return vol_arr.shape[2] // 2

def mid_axial_index(vol_arr):
    # pick mid Z slice
    return vol_arr.shape[0] // 2



# === Cell 2A: Robust series discovery (overrides) ===
import pydicom
from collections import defaultdict
import math

def _series_score(first_file):
    """Return (axial_hint, series_desc_hint) for scoring."""
    axial_hint = 0
    desc_hint  = 0
    try:
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, specific_tags=[
            "SeriesDescription","ImageOrientationPatient"
        ])
        sd = (ds.get("SeriesDescription") or "").upper()
        if "AX" in sd or "AXIAL" in sd:
            axial_hint = 1
        # Orientation-based axial hint (normal ~ Z axis)
        iop = ds.get("ImageOrientationPatient")
        if iop and len(iop)==6:
            # row, col direction cosines -> normal
            r = np.array(iop[:3], dtype=float); c = np.array(iop[3:], dtype=float)
            n = np.cross(r, c)
            if abs(n[2]) >= 0.8:  # slice normal mostly along Z
                axial_hint = max(axial_hint, 1)
        # prefer typical CT image types
        if "BONE" in sd or "STD" in sd or "SOFT" in sd:
            desc_hint = 1
    except Exception:
        pass
    return axial_hint, desc_hint

def _gather_series(study_folder: Path, max_depth=2):
    """Search study folder (and subfolders) for DICOM series; group by SeriesInstanceUID."""
    dirs = [study_folder]
    if max_depth >= 1:
        dirs += [d for d in study_folder.glob("*") if d.is_dir()]
    if max_depth >= 2:
        dirs += [d for d in study_folder.glob("*/*") if d.is_dir()]
    series_map = defaultdict(list)  # uid -> list of files
    first_file_for_uid = {}
    for d in dirs:
        # List .dcm files directly within this dir (no deeper to keep perf acceptable)
        dcm_files = sorted([str(x) for x in d.glob("*.dcm")])
        if not dcm_files:
            continue
        # Group by SeriesInstanceUID
        for fp in dcm_files:
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, specific_tags=["SeriesInstanceUID"])
                uid = str(ds.get("SeriesInstanceUID"))
                if uid:
                    series_map[uid].append(fp)
                    if uid not in first_file_for_uid:
                        first_file_for_uid[uid] = fp
            except Exception:
                continue
    # rank UIDs
    ranked = []
    for uid, files in series_map.items():
        axial_hint, desc_hint = _series_score(first_file_for_uid[uid])
        ranked.append((len(files), axial_hint, desc_hint, uid, sorted(files)))
    ranked.sort(reverse=True)  # prefer more slices, axial, better desc
    return ranked  # list of tuples

def load_best_series_any(study_folder: Path):
    """
    Return (sitk_img, np_arr, files_used, uid) for the best series we can find.
    Tries direct SITK first; if none, uses pydicom grouping fallback.
    """
    # Try SITK on study root
    uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(study_folder))
    best_tuple = None
    if uids:
        cand = []
        for uid in uids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(study_folder), uid)
            # simple axial score by SeriesDescription (read first file)
            axial_hint, desc_hint = _series_score(files[0])
            cand.append((len(files), axial_hint, desc_hint, uid, files))
        cand.sort(reverse=True)
        best_tuple = cand[0]

    if best_tuple is None:
        # Fallback: scan subfolders, group by UID via pydicom
        ranked = _gather_series(study_folder, max_depth=2)
        if ranked:
            best_tuple = ranked[0]

    if best_tuple is None:
        return None, None, None, None

    nfiles, axial_hint, desc_hint, uid, files = best_tuple
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    print(f"[LOAD] Series UID={uid} | slices={nfiles} | axial={axial_hint} | desc_hint={desc_hint}")
    return img, arr, files, uid



# === Cell 3: Find a study with segmentation and visualize axial & sagittal overlays ===

def find_first_study_with_seg(study_dirs, seg_map):
    for p in study_dirs:
        if p.name in seg_map:
            return p, seg_map[p.name]
    return None, None

study_dir, seg_path = find_first_study_with_seg(study_dirs, seg_map)
assert study_dir is not None, "No study with segmentation found. Check that 'segmentations' were added in 'Add data'."

print(f"[PICK] study_id: {study_dir.name}")
series_dir = pick_axial_series(study_dir)
assert series_dir is not None, f"No axial series found under {study_dir}"

print(f"[LOAD] DICOM series: {series_dir}")
ct_img, ct_arr = load_dicom_volume(series_dir)
assert ct_img is not None, "Failed to load DICOM volume"

print(f"[LOAD] NIfTI seg: {seg_path.name}")
seg_img_mov, seg_arr_mov = load_seg_nifti(seg_path)

# Resample segmentation into CT geometry if needed
same_geom = (ct_img.GetSize()==seg_img_mov.GetSize() and
             np.allclose(ct_img.GetSpacing(), seg_img_mov.GetSpacing(), atol=1e-3) and
             tuple(ct_img.GetDirection())==tuple(seg_img_mov.GetDirection()))
if not same_geom:
    seg_img_ct = resample_like(seg_img_mov, ct_img, is_label=True)
    seg_arr = sitk.GetArrayFromImage(seg_img_ct)
    print("[INFO] Resampled segmentation to CT geometry.")
else:
    seg_arr = seg_arr_mov

print(f"[SHAPE] CT: {ct_arr.shape} | SEG: {seg_arr.shape}  (Z,Y,X)")

# Choose display planes
z_ax = mid_axial_index(ct_arr)
x_sag = mid_sagittal_index(ct_arr)

# Prepare slices
ax_ct  = ct_arr[z_ax, :, :]
ax_lbl = seg_arr[z_ax, :, :]
sag_ct  = ct_arr[:, :, x_sag]
sag_lbl = seg_arr[:, :, x_sag]

# Window images and overlays
ax_vis  = window_image(ax_ct, center=400, width=1800)  # bone-friendly
sag_vis = window_image(sag_ct, center=400, width=1800)

ax_ov  = overlay_segmentation(ax_vis,  ax_lbl)
sag_ov = overlay_segmentation(sag_vis, sag_lbl)

# Plot
fig, axs = plt.subplots(1,2, figsize=(10,4.5))
axs[0].imshow(ax_ov);  axs[0].set_title(f"Axial mid-slice (z={z_ax})")
axs[1].imshow(sag_ov); axs[1].set_title(f"Sagittal mid-column (x={x_sag})")
for a in axs: a.axis('off')
plt.suptitle(f"Study: {study_dir.name}\nVertebra segmentation overlay (labels 1–7 = C1..C7)")
plt.show()

# Quick label summary for this study
vals, cnts = np.unique(seg_arr, return_counts=True)
present = {int(v): int(c) for v,c in zip(vals, cnts) if v>0}
print("[EDA] Labels present in segmentation (value: voxel_count) ->", present)



# === Cell 3A: Find a working CT+seg pair, resample seg, and show overlays ===

def find_working_pair_with_seg(study_dirs, seg_map, max_trials=50):
    trials = 0
    for p in study_dirs:
        if p.name not in seg_map: 
            continue
        trials += 1
        if trials > max_trials:
            break
        try:
            ct_img, ct_arr, files, uid = load_best_series_any(p)
            if ct_img is None: 
                print(f"[SKIP] No series found for {p.name}")
                continue
            seg_img_mov, seg_arr_mov = load_seg_nifti(seg_map[p.name])
            # resample seg to CT geometry if needed
            same_geom = (ct_img.GetSize()==seg_img_mov.GetSize() and
                         np.allclose(ct_img.GetSpacing(), seg_img_mov.GetSpacing(), atol=1e-3) and
                         tuple(ct_img.GetDirection())==tuple(seg_img_mov.GetDirection()))
            if not same_geom:
                seg_img_ct = resample_like(seg_img_mov, ct_img, is_label=True)
                seg_arr = sitk.GetArrayFromImage(seg_img_ct)
                print(f"[INFO] Resampled seg -> CT geometry for {p.name}")
            else:
                seg_arr = seg_arr_mov
            # sanity: must have some C1..C7 labels
            if not np.any(np.isin(seg_arr, np.arange(1,8))):
                print(f"[SKIP] No cervical labels (1..7) in seg for {p.name}")
                continue
            return p, ct_img, ct_arr, seg_arr
        except Exception as e:
            print(f"[WARN] {p.name} failed: {e}")
            continue
    return None, None, None, None

study_dir2, ct_img2, ct_arr2, seg_arr2 = find_working_pair_with_seg(study_dirs, seg_map, max_trials=120)
assert study_dir2 is not None, "Could not find a study with both CT and cervical segmentation after 120 trials."

print(f"[OK] Using study: {study_dir2.name} | CT shape={ct_arr2.shape} | SEG shape={seg_arr2.shape}")

# Choose display planes
z_ax = mid_axial_index(ct_arr2)
x_sag = mid_sagittal_index(ct_arr2)

# Prepare slices
ax_ct  = ct_arr2[z_ax, :, :]
ax_lbl = seg_arr2[z_ax, :, :]
sag_ct  = ct_arr2[:, :, x_sag]
sag_lbl = seg_arr2[:, :, x_sag]

# Window images and overlays
ax_vis  = window_image(ax_ct, center=400, width=1800)
sag_vis = window_image(sag_ct, center=400, width=1800)

ax_ov  = overlay_segmentation(ax_vis,  ax_lbl)
sag_ov = overlay_segmentation(sag_vis, sag_lbl)

# Plot
fig, axs = plt.subplots(1,2, figsize=(10,4.5))
axs[0].imshow(ax_ov);  axs[0].set_title(f"Axial mid-slice (z={z_ax})")
axs[1].imshow(sag_ov); axs[1].set_title(f"Sagittal mid-column (x={x_sag})")
for a in axs: a.axis('off')
plt.suptitle(f"Study: {study_dir2.name}\nVertebra segmentation overlay (labels 1–7 = C1..C7)")
plt.show()

vals, cnts = np.unique(seg_arr2, return_counts=True)
present = {int(v): int(c) for v,c in zip(vals, cnts) if v>0}
print("[EDA] Labels present (value: voxel_count) ->", present)



# Recompute windows & overlays after the patch
ax_vis  = window_image(ax_ct,  center=400, width=1800)
sag_vis = window_image(sag_ct, center=400, width=1800)

ax_ov  = overlay_segmentation(ax_vis,  ax_lbl)
sag_ov = overlay_segmentation(sag_vis, sag_lbl)

fig, axs = plt.subplots(1,2, figsize=(10,4.5))
axs[0].imshow(ax_ov);  axs[0].set_title(f"Axial mid-slice (z={z_ax})")
axs[1].imshow(sag_ov); axs[1].set_title(f"Sagittal mid-column (x={x_sag})")
for a in axs: a.axis('off')
plt.suptitle(f"Study: {study_dir2.name}\nVertebra segmentation overlay (labels 1–7=C1..C7; 8–19=T1..T12)")
plt.show()



# === Cell 3P: Aspect-correct, orientation-aware display ===

def show_overlay(img2d, lbl2d, spacing_xy, title="", flip_ud=False, flip_lr=False):
    """
    img2d: 2D numpy (float in [0,1] recommended); lbl2d: int labels (same HW)
    spacing_xy: (sy, sx) in mm for the displayed plane
    flip_ud, flip_lr: visual-only flips to correct orientation
    """
    v = img2d
    m = lbl2d
    if flip_ud: v, m = np.flipud(v), np.flipud(m)
    if flip_lr: v, m = np.fliplr(v), np.fliplr(m)

    ov = overlay_segmentation(v, m)
    sy, sx = float(spacing_xy[0]), float(spacing_xy[1])
    H, W = v.shape
    # extent maps pixel indices to physical mm so aspect is preserved
    extent = [0, W * sx, H * sy, 0]  # x0, x1, y0, y1

    fig, ax = plt.subplots(figsize=(6, 6 * (H*sy)/(W*sx)))
    ax.imshow(ov, extent=extent, interpolation='nearest')
    ax.set_aspect('equal')  # preserve mm aspect
    ax.set_title(title)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Convenience wrappers for axial & sagittal using ct_img2 spacing:
# ITK spacing order is (sx, sy, sz); array orders we show are:
#   axial: (Y,X) -> (sy, sx)
#   sagittal: (Z,Y) -> (sz, sy)
def show_axial(ax_img, ax_lbl, ct_img):
    sx, sy, sz = ct_img.GetSpacing()
    show_overlay(ax_img, ax_lbl, spacing_xy=(sy, sx),
                 title="Axial mid-slice (aspect-correct)")

def show_sagittal(sag_img, sag_lbl, ct_img, flip_upside_down=True):
    sx, sy, sz = ct_img.GetSpacing()
    # Many DICOMs render sagittal with superior at bottom; flip_ud=True fixes that visually
    show_overlay(sag_img, sag_lbl, spacing_xy=(sz, sy),
                 title="Sagittal mid-column (aspect-correct)",
                 flip_ud=bool(flip_upside_down), flip_lr=False)



# Re-window just in case
ax_vis  = window_image(ax_ct,  center=400, width=1800)
sag_vis = window_image(sag_ct, center=400, width=1800)

show_axial(ax_vis,  ax_lbl,  ct_img2)
show_sagittal(sag_vis, sag_lbl, ct_img2, flip_upside_down=True)  # set False if your site already shows “upright”



# === Cell 4: Batch EDA — coverage stats + small overlay gallery ===

def study_has_cervical(seg_arr):
    return np.any(np.isin(seg_arr, np.arange(1,8)))  # 1..7 = C1..C7

def overlay_panel(study_dir, seg_path, max_panels=4):
    try:
        series_dir = pick_axial_series(study_dir)
        ct_img, ct_arr = load_dicom_volume(series_dir)
        seg_img_mov, seg_arr_mov = load_seg_nifti(seg_path)
        # resample if needed
        same_geom = (ct_img.GetSize()==seg_img_mov.GetSize() and
                     np.allclose(ct_img.GetSpacing(), seg_img_mov.GetSpacing(), atol=1e-3) and
                     tuple(ct_img.GetDirection())==tuple(seg_img_mov.GetDirection()))
        seg_arr_ct = sitk.GetArrayFromImage(resample_like(seg_img_mov, ct_img, True)) if not same_geom else seg_arr_mov

        z = mid_axial_index(ct_arr); x = mid_sagittal_index(ct_arr)
        ax_vis  = window_image(ct_arr[z]); ax_lbl  = seg_arr_ct[z]
        sag_vis = window_image(ct_arr[:,:,x]); sag_lbl = seg_arr_ct[:,:,x]
        ax_ov  = overlay_segmentation(ax_vis,  ax_lbl)
        sag_ov = overlay_segmentation(sag_vis, sag_lbl)

        fig, axs = plt.subplots(1,2, figsize=(8.5,3.6))
        axs[0].imshow(ax_ov);  axs[0].set_title(f"Axial z={z}")
        axs[1].imshow(sag_ov); axs[1].set_title(f"Sagittal x={x}")
        for a in axs: a.axis('off')
        plt.suptitle(f"Study {study_dir.name} — Vertebra segmentation overlay")
        plt.tight_layout(); plt.show()
        return True
    except Exception as e:
        print(f"[WARN] Failed panel for {study_dir.name}: {e}")
        return False

# Coverage scan (first 300 studies for speed)
hits = 0; cerv_hits = 0
examples = 0
for p in study_dirs[:300]:
    if p.name not in seg_map: continue
    seg_img, seg_arr = load_seg_nifti(seg_map[p.name])
    hits += 1
    if study_has_cervical(seg_arr): 
        cerv_hits += 1
        if examples < 3:
            ok = overlay_panel(p, seg_map[p.name])
            if ok: examples += 1

print(f"[EDA] In first 300 studies: {hits} have segmentation; {cerv_hits} contain C1–C7 labels.")



# === Cell 4A: Gallery (up to 4 studies) with crisp overlays ===
picked = [study_dir2.name]
shown = 0
for p in study_dirs:
    if p.name in picked or p.name not in seg_map:
        continue
    try:
        ct_img, ct_arr, files, uid = load_best_series_any(p)
        if ct_img is None: 
            continue
        seg_img_mov, seg_arr_mov = load_seg_nifti(seg_map[p.name])
        same_geom = (ct_img.GetSize()==seg_img_mov.GetSize() and
                     np.allclose(ct_img.GetSpacing(), seg_img_mov.GetSpacing(), atol=1e-3) and
                     tuple(ct_img.GetDirection())==tuple(seg_img_mov.GetDirection()))
        seg_arr = sitk.GetArrayFromImage(resample_like(seg_img_mov, ct_img, True)) if not same_geom else seg_arr_mov
        if not np.any(np.isin(seg_arr, np.arange(1,8))):
            continue

        z = mid_axial_index(ct_arr); x = mid_sagittal_index(ct_arr)
        ax_vis  = window_image(ct_arr[z]); ax_lbl  = seg_arr[z]
        sag_vis = window_image(ct_arr[:,:,x]); sag_lbl = seg_arr[:,:,x]
        ax_ov  = overlay_segmentation(ax_vis,  ax_lbl)
        sag_ov = overlay_segmentation(sag_vis, sag_lbl)

        fig, axs = plt.subplots(1,2, figsize=(8.8,3.6))
        axs[0].imshow(ax_ov);  axs[0].set_title(f"Axial z={z}")
        axs[1].imshow(sag_ov); axs[1].set_title(f"Sagittal x={x}")
        for a in axs: a.axis('off')
        plt.suptitle(f"Study {p.name} — Vertebra segmentation overlay")
        plt.tight_layout(); plt.show()

        picked.append(p.name); shown += 1
        if shown >= 3: break
    except Exception as e:
        print(f"[WARN] Gallery skip {p.name}: {e}")
        continue



# === Cell 5: Build per-study label presence table (C1..C7) ===

rows = []
for p in study_dirs[:600]:  # sample subset for speed; expand later
    sid = p.name
    has_seg = sid in seg_map
    row = {"study_id": sid, "has_seg": has_seg}
    if has_seg:
        try:
            seg_img, seg_arr = load_seg_nifti(seg_map[sid])
            for k in range(1,8):
                row[f"C{k}"] = int(np.any(seg_arr==k))
        except Exception as e:
            row.update({f"C{k}": -1 for k in range(1,8)})
    else:
        row.update({f"C{k}": 0 for k in range(1,8)})
    rows.append(row)

df_cov = pd.DataFrame(rows)
print("[EDA] per-study cervical label presence:")
display(df_cov.head(10))

print("[EDA] counts:\n", df_cov[[f"C{k}" for k in range(1,8)]].sum())



# Filter to “eligible” studies for measurement
eligible = []
for p in study_dirs:
    if p.name not in seg_map: 
        continue
    seg_img, seg_arr = load_seg_nifti(seg_map[p.name])
    if np.any(np.isin(seg_arr, np.arange(1,8))):  # any of C1..C7 present
        eligible.append(p)
print(f"[FILTER] Eligible studies with cervical labels: {len(eligible)}")



def disc_mid_z_fallback(seg_arr, upper_id, lower_id):
    up = (seg_arr==upper_id); lo = (seg_arr==lower_id)
    if up.any() and lo.any():
        zu = int(np.median(np.argwhere(up)[:,0])); zl = int(np.median(np.argwhere(lo)[:,0]))
        return int(round((zu+zl)/2))
    # fallback: if only one present, use that vertebra's median Z
    if up.any():
        return int(np.median(np.argwhere(up)[:,0]))
    if lo.any():
        return int(np.median(np.argwhere(lo)[:,0]))
    return None

# drop-in replacement inside Cell 5A:
def disc_mid_z(seg_arr, upper_id, lower_id):
    return disc_mid_z_fallback(seg_arr, upper_id, lower_id)



# === Cell 5A: Per-level localization utilities ===
CERV_IDS = np.arange(1, 8)  # 1..7 = C1..C7 in RSNA2022

def level_presence(seg_arr):
    pres = {int(k): bool((seg_arr==k).any()) for k in CERV_IDS}
    return pres

def centroid_zy(mask3d):
    # returns (z,y) centroid in voxel coordinates for a 3D boolean mask
    idx = np.argwhere(mask3d)
    if idx.size == 0: return None
    z = int(np.median(idx[:,0])); y = int(np.median(idx[:,1]))
    return z, y

def disc_mid_z(seg_arr, upper_id, lower_id):
    # mid-z plane between two adjacent vertebra centroids (robust median)
    up = (seg_arr==upper_id)
    lo = (seg_arr==lower_id)
    cu = centroid_zy(up); cl = centroid_zy(lo)
    if cu is None or cl is None: return None
    return int(round((cu[0] + cl[0]) / 2))

def midsag_x_from_mask(seg_arr, lvl_id):
    # choose x column that traverses the vertebra mask centerline (robust median over occupied x)
    xs = np.argwhere(seg_arr==lvl_id)[:,2] if (seg_arr==lvl_id).any() else None
    if xs is None or xs.size==0: return seg_arr.shape[2]//2
    return int(np.median(xs))

# Build a table of available C2..C7 levels with suggested planes
rows = []
for lid in range(2,8):  # C2..C7 planes defined by vertebra C{lid} and C{lid+1} (use prior if missing)
    rows.append(lid)

perlevel = []
for lid in range(2,8):  # define disc planes C2/3 ... C6/7
    upper, lower = lid, min(lid+1, 7)
    z_mid = disc_mid_z(seg_arr2, upper, lower) if lower!=upper else None
    x_mid = midsag_x_from_mask(seg_arr2, upper)
    perlevel.append({"level": f"C{lid}/C{lower}", "upper":upper, "lower":lower,
                     "z_mid": z_mid, "x_mid": x_mid})
perlevel_df = pd.DataFrame(perlevel)
print("[LOC] Proposed planes (nan if a vertebra label is missing):")
display(perlevel_df)



# === Cell 6B: Mask-guided APD & Torg (robust) ===
BONE_TH = 200.0  # lower a bit for soft-tissue kernels; try 180–250 if needed
Z_NEIGHBOR = 3   # median over ±3 slices around z_mid

def _posterior_body_from_mask(seg_band_zy, upper_id):
    """
    seg_band_zy: (k, Y) int labels for a sagittal band (z-neighborhood at fixed x_mid)
    Return (y_vb_ant, y_vb_post) from the vertebral BODY mask (upper_id) projected along z.
    """
    band = (seg_band_zy == int(upper_id)).astype(np.uint8)  # (k,Y)
    proj = band.max(axis=0)  # (Y,)
    ys = np.where(proj > 0)[0]
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max())

def _lamina_anterior_y(ct_band_zy, y_start_post):
    """
    Find anterior cortex of posterior elements: first contiguous bone cluster posterior to y_start_post.
    ct_band_zy: (k,Y) HU array; we binarize > BONE_TH and OR over k.
    """
    bone = (ct_band_zy > BONE_TH)
    prof = bone.any(axis=0).astype(np.uint8)  # (Y,)
    prof[:max(0, y_start_post+1)] = 0
    ys = np.where(prof > 0)[0]
    if ys.size == 0:
        return None
    return int(ys[0])

def measure_level_APD_Torg_mask_guided(ct_arr, ct_img, seg_arr, upper_id, z_mid, x_mid,
                                       z_neighborhood=Z_NEIGHBOR):
    """
    Robust APD (mm) and Torg using vertebra mask guidance on a sagittal band.
    """
    if z_mid is None or x_mid is None:
        return None
    sx, sy, sz = ct_img.GetSpacing()  # (sx, sy, sz) -> use sy for AP distance in sagittal (Y)
    z0 = max(0, int(z_mid) - z_neighborhood)
    z1 = min(ct_arr.shape[0]-1, int(z_mid) + z_neighborhood)

    # build sagittal bands (k,Y)
    ct_band = ct_arr[z0:z1+1, :, int(x_mid)].astype(np.float32)
    seg_band = seg_arr[z0:z1+1, :, int(x_mid)].astype(np.int32)

    vb = _posterior_body_from_mask(seg_band, upper_id)
    if vb is None:
        return None
    y_vb_ant, y_vb_post = vb

    y_lam_ant = _lamina_anterior_y(ct_band, y_vb_post)
    if y_lam_ant is None or y_lam_ant <= y_vb_post:
        return None

    apd_mm   = (y_lam_ant - y_vb_post) * float(sy)
    vb_ap_mm = max(0.0, (y_vb_post - y_vb_ant) * float(sy))
    torg     = (apd_mm / vb_ap_mm) if vb_ap_mm > 0 else np.nan
    return {"APD_mm": float(apd_mm), "Torg": float(torg)}

def severity_from_apd_torg(apd_mm, torg):
    # Severe <10 mm; Moderate 10–13 mm; supportive Torg <0.8 (Pavlov/Torg). 
    if apd_mm < 10.0: sev = 2
    elif apd_mm < 13.0: sev = 1
    else: sev = 0
    if not np.isnan(torg) and torg < 0.8 and 9.5 <= apd_mm <= 13.5:
        sev = min(2, sev+1)
    return sev

# Recompute measurements for this study using mask-guided method
meas_rows = []
for r in perlevel:
    z_mid, x_mid, up = r["z_mid"], r["x_mid"], r["upper"]
    if z_mid is None:
        meas_rows.append({**r, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    m = measure_level_APD_Torg_mask_guided(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-v2] Per-level measurements (mask-guided):")
display(meas_df)



# === Cell 6C: Axial-plane mask-guided APD & Torg ===
BONE_TH = 200.0           # soften if needed (180–250 works across kernels)
X_BAND  = 3               # median over x ∈ [x_mid±X_BAND]
Z_NEIGH = 1               # small z jitter around disc plane if needed

def _axial_body_edges(ax_seg2d, vertebra_id):
    """
    From axial segmentation at z=z_mid: for the given vertebra label,
    return (x_min, x_max, y_min, y_max) bounding box and the posterior
    body cortex y index (y_vb_post) estimated from the label mask.
    """
    mask = (ax_seg2d == int(vertebra_id))
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    # posterior body cortex approximated as the max Y within the vertebral body mask
    y_vb_post = int(y_max)
    return (x_min, x_max, y_min, y_max, y_vb_post)

def _axial_lamina_anterior_y(ax_ct2d, y_start_post, x_col):
    """
    Scan posteriorly (in +Y) along the column x=x_col on axial CT,
    find the first bone cluster (> BONE_TH) posterior to y_start_post.
    Returns y_lam_ant or None.
    """
    col = ax_ct2d[:, int(x_col)]
    bone = (col > BONE_TH).astype(np.uint8)
    bone[:max(0, y_start_post+1)] = 0  # zero anterior to/posterior body cortex
    ys = np.where(bone > 0)[0]
    return int(ys[0]) if ys.size else None

def measure_level_APD_Torg_axial(ct_arr, ct_img, seg_arr, up_id, z_mid, x_mid,
                                 x_band=X_BAND, z_neigh=Z_NEIGH):
    """
    Compute APD (mm) and Torg on the **axial** slice near z_mid.
    AP = Y-axis in axial (array shape (Y,X)); lateral = X-axis.
    We use a small median over x ∈ [x_mid ± x_band] and z ∈ [z_mid ± z_neigh].
    """
    if z_mid is None or x_mid is None:
        return None

    sx, sy, sz = ct_img.GetSpacing()   # axial pixels: (sy, sx) → AP uses sy
    z0 = max(0, int(z_mid) - z_neigh)
    z1 = min(ct_arr.shape[0]-1, int(z_mid) + z_neigh)
    x0 = max(0, int(x_mid) - x_band)
    x1 = min(ct_arr.shape[2]-1, int(x_mid) + x_band)

    apd_vals, torg_vals = [], []

    for z in range(z0, z1+1):
        ax_ct  = ct_arr[z].astype(np.float32)     # (Y,X)
        ax_seg = seg_arr[z].astype(np.int32)      # (Y,X)

        body_box = _axial_body_edges(ax_seg, up_id)
        if body_box is None:
            continue
        x_min, x_max, y_min, y_max, y_vb_post = body_box

        # VB anterior edge (rough): y_min from mask
        y_vb_ant = int(y_min)

        # scan across a small lateral band around x_mid
        y_lams = []
        for x in range(x0, x1+1):
            y_lam = _axial_lamina_anterior_y(ax_ct, y_vb_post, x)
            if y_lam is not None and y_lam > y_vb_post:
                y_lams.append(y_lam)

        if not y_lams:
            continue

        y_lam_ant = int(np.median(y_lams))
        apd_mm    = max(0.0, (y_lam_ant - y_vb_post) * float(sy))
        vb_ap_mm  = max(0.0, (y_vb_post - y_vb_ant) * float(sy))
        if vb_ap_mm <= 0:
            continue

        torg = apd_mm / vb_ap_mm
        apd_vals.append(apd_mm); torg_vals.append(torg)

    if not apd_vals:
        return None

    apd  = float(np.median(apd_vals))
    torg = float(np.median(torg_vals)) if torg_vals else np.nan
    return {"APD_mm": apd, "Torg": torg}

def severity_from_apd_torg(apd_mm, torg):
    # Severe < 10 mm; Moderate 10–13 mm; Torg < 0.8 boosts near-boundaries
    if apd_mm < 10.0: sev = 2
    elif apd_mm < 13.0: sev = 1
    else: sev = 0
    if not np.isnan(torg) and torg < 0.8 and 9.5 <= apd_mm <= 13.5:
        sev = min(2, sev+1)
    return sev

# Recompute measurements using axial method
meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid, x_mid, up = r["z_mid"], r["x_mid"], r["upper"]
    if z_mid is None:
        meas_rows.append({**r, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    m = measure_level_APD_Torg_axial(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-axial] Per-level measurements (mm) and severity:")
display(meas_df)

# === Patch: safe fallbacks for z_mid / x_mid when NaN ===
def safe_zmid(seg_arr, z_mid, upper_id, lower_id):
    # if z_mid is valid
    if z_mid == z_mid:  # not NaN
        return int(round(z_mid))
    # fallback to median Z of whichever vertebra label exists
    for vid in (int(upper_id), int(lower_id)):
        m = (seg_arr == vid)
        if m.any():
            return int(np.median(np.argwhere(m)[:,0]))
    # last resort: center slice
    return seg_arr.shape[0] // 2

def safe_xmid(seg_arr, x_mid, upper_id):
    if x_mid == x_mid:  # not NaN
        return int(round(x_mid))
    xs = np.argwhere(seg_arr == int(upper_id))[:,2] if (seg_arr == int(upper_id)).any() else None
    return int(np.median(xs)) if xs is not None and xs.size else seg_arr.shape[2] // 2

# Re-run the measurement loop with safe fallbacks (uses measure_level_APD_Torg_axial from 6C)
meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid = safe_zmid(seg_arr2, r["z_mid"], r["upper"], r["lower"])
    x_mid = safe_xmid(seg_arr2, r["x_mid"], r["upper"])
    up    = int(r["upper"])

    m = measure_level_APD_Torg_axial(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid,
                          "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-axial] Per-level measurements with safe fallbacks:")
display(meas_df)



# === Cell 6D: Axial mask-guided APD/Torg with body-arch separation ===
# Uses vertebra segmentation to isolate the vertebral BODY, then scans posteriorly to lamina/spinous cortex
# on the union of all bony labels. Median across a small lateral band for robustness.

import numpy as np
from scipy import ndimage as ndi

BONE_TH  = 180.0  # only used if you later want HU fallback; segmentation union is primary here
X_BAND   = 5      # median over x ∈ [x_mid ± X_BAND]
Z_NEIGH  = 1      # small z jitter around disc plane if needed
OPEN_SZ  = 5      # morphological opening size to remove thin posterior elements

def _axial_body_from_seg(ax_seg2d, vertebra_id):
    """
    Return (y_vb_ant, y_vb_post, body_bbox) for the VERTEBRAL BODY only.
    Strategy: binary opening to delete thin arches; if still connected, keep anterior 65% of mask.
    """
    mask = (ax_seg2d == int(vertebra_id))
    if not mask.any(): 
        return None

    opened = ndi.binary_opening(mask, structure=np.ones((OPEN_SZ, OPEN_SZ)))
    body = opened.copy()
    if not body.any():
        # Fallback: cut posterior 35% of the original vertebra label
        ys = np.where(mask)[0]
        y_min, y_max = ys.min(), ys.max()
        y_cut = int(y_min + 0.65*(y_max - y_min))
        body = mask.copy()
        body[y_cut+1:, :] = 0

    ys_body, xs_body = np.where(body)
    if ys_body.size == 0:
        return None
    y_vb_ant = int(ys_body.min())
    y_vb_post = int(ys_body.max())
    x_min, x_max = int(xs_body.min()), int(xs_body.max())
    return y_vb_ant, y_vb_post, (x_min, x_max, int(ys_body.min()), int(ys_body.max()))

def _axial_lamina_anterior_y_from_seg(ax_seg2d, y_start_post, x_range):
    """
    First posterior bony cortex behind the vertebral body:
    use the union of all labels (>0) so lamina/spinous processes are included.
    """
    bone_union = (ax_seg2d > 0).astype(np.uint8)
    y_list = []
    for x in x_range:
        col = bone_union[:, int(x)].copy()
        col[:max(0, y_start_post+1)] = 0  # zero anterior to (and at) posterior body cortex
        ys = np.where(col > 0)[0]
        if ys.size:
            y_list.append(int(ys[0]))
    if not y_list:
        return None
    return int(np.median(y_list))

def measure_level_APD_Torg_axial(ct_arr, ct_img, seg_arr, up_id, z_mid, x_mid,
                                 x_band=X_BAND, z_neigh=Z_NEIGH):
    """
    Compute APD (mm) and Torg on the axial plane near z_mid using segmentation only.
    AP is the Y-axis in axial; mm scaling uses sy from spacing.
    """
    if z_mid is None or x_mid is None:
        return None

    sx, sy, sz = ct_img.GetSpacing()   # axial px spacing: (sy, sx)
    z0 = max(0, int(z_mid) - z_neigh)
    z1 = min(ct_arr.shape[0]-1, int(z_mid) + z_neigh)
    x0 = max(0, int(x_mid) - x_band)
    x1 = min(ct_arr.shape[2]-1, int(x_mid) + x_band)

    apd_vals, torg_vals = [], []

    for z in range(z0, z1+1):
        ax_seg = seg_arr[z].astype(np.int32)

        vb = _axial_body_from_seg(ax_seg, up_id)
        if vb is None:
            continue
        y_vb_ant, y_vb_post, _bbox = vb

        # posterior bony cortex (lamina/spinous) from union of all labels
        y_lam_ant = _axial_lamina_anterior_y_from_seg(ax_seg, y_vb_post, range(x0, x1+1))
        if y_lam_ant is None or y_lam_ant <= y_vb_post:
            continue

        apd_mm   = (y_lam_ant - y_vb_post) * float(sy)
        vb_ap_mm = max(0.0, (y_vb_post - y_vb_ant) * float(sy))
        if vb_ap_mm <= 0: 
            continue

        torg = apd_mm / vb_ap_mm
        apd_vals.append(apd_mm)
        torg_vals.append(torg)

    if not apd_vals:
        return None
    apd  = float(np.median(apd_vals))
    torg = float(np.median(torg_vals)) if torg_vals else np.nan
    return {"APD_mm": apd, "Torg": torg}

def severity_from_apd_torg(apd_mm, torg):
    # Severe <10 mm; Moderate 10–13 mm; Torg <0.8 supports up-binning near boundaries
    if apd_mm < 10.0: sev = 2
    elif apd_mm < 13.0: sev = 1
    else: sev = 0
    if not np.isnan(torg) and torg < 0.8 and 9.5 <= apd_mm <= 13.5:
        sev = min(2, sev+1)
    return sev

# --- Recompute with safe fallbacks for z/x mids ---
def safe_zmid(seg_arr, z_mid, upper_id, lower_id):
    if z_mid == z_mid:  # not NaN
        return int(round(z_mid))
    for vid in (int(upper_id), int(lower_id)):
        m = (seg_arr == vid)
        if m.any():
            return int(np.median(np.argwhere(m)[:,0]))
    return seg_arr.shape[0] // 2

def safe_xmid(seg_arr, x_mid, upper_id):
    if x_mid == x_mid:
        return int(round(x_mid))
    m = (seg_arr == int(upper_id))
    if m.any():
        xs = np.argwhere(m)[:,2]
        return int(np.median(xs))
    return seg_arr.shape[2] // 2

meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid = safe_zmid(seg_arr2, r["z_mid"], r["upper"], r["lower"])
    x_mid = safe_xmid(seg_arr2, r["x_mid"], r["upper"])
    up    = int(r["upper"])
    m = measure_level_APD_Torg_axial(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-axial v2] Per-level measurements with body/arch separation:")
display(meas_df)



# === Patch: safe fallbacks for z_mid / x_mid when NaN ===
def safe_zmid(seg_arr, z_mid, upper_id, lower_id):
    # if z_mid is valid
    if z_mid == z_mid:  # not NaN
        return int(round(z_mid))
    # fallback to median Z of whichever vertebra label exists
    for vid in (int(upper_id), int(lower_id)):
        m = (seg_arr == vid)
        if m.any():
            return int(np.median(np.argwhere(m)[:,0]))
    # last resort: center slice
    return seg_arr.shape[0] // 2

def safe_xmid(seg_arr, x_mid, upper_id):
    if x_mid == x_mid:  # not NaN
        return int(round(x_mid))
    xs = np.argwhere(seg_arr == int(upper_id))[:,2] if (seg_arr == int(upper_id)).any() else None
    return int(np.median(xs)) if xs is not None and xs.size else seg_arr.shape[2] // 2

# Re-run the measurement loop with safe fallbacks (uses measure_level_APD_Torg_axial from 6C)
meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid = safe_zmid(seg_arr2, r["z_mid"], r["upper"], r["lower"])
    x_mid = safe_xmid(seg_arr2, r["x_mid"], r["upper"])
    up    = int(r["upper"])

    m = measure_level_APD_Torg_axial(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid,
                          "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-axial] Per-level measurements with safe fallbacks:")
display(meas_df)



# === QC (axial) — draw body posterior, lamina anterior, and APD on the chosen level ===
def qc_axial_level_v2(ct_arr, ct_img, seg_arr, level_row, x_band=X_BAND):
    z_mid, x_mid, up = int(level_row["z_mid"]), int(level_row["x_mid"]), int(level_row["upper"])
    sx, sy, sz = ct_img.GetSpacing()
    ax_ct  = ct_arr[z_mid].astype(np.float32)
    ax_seg = seg_arr[z_mid].astype(np.int32)

    vb = _axial_body_from_seg(ax_seg, up)
    if vb is None:
        print("No vertebral BODY on this slice.")
        return
    y_vb_ant, y_vb_post, _ = vb

    x0 = max(0, x_mid - x_band); x1 = min(ax_ct.shape[1]-1, x_mid + x_band)
    y_lam_ant = _axial_lamina_anterior_y_from_seg(ax_seg, y_vb_post, range(x0, x1+1))

    img = window_image(ax_ct, 400, 1800)
    ov  = overlay_segmentation(img, ax_seg)

    H,W = img.shape
    extent = [0, W*sx, H*sy, 0]
    fig, ax = plt.subplots(figsize=(6, 6*(H*sy)/(W*sx)))
    ax.imshow(ov, extent=extent); ax.set_aspect('equal'); ax.axis('off')

    # central column
    x_mm = x_mid * sx
    ax.axvline(x_mm, color="white", lw=1.2, linestyle="--")
    # posterior/anterior body, lamina
    ax.hlines([y_vb_ant*sy, y_vb_post*sy], x_mm-5, x_mm+5, colors=["cyan","yellow"], lw=2)
    if y_lam_ant is not None:
        ax.hlines([y_lam_ant*sy], x_mm-5, x_mm+5, colors=["red"], lw=2)
        apd_mm = (y_lam_ant - y_vb_post) * sy
        ax.text(x_mm+6, (y_vb_post*sy + y_lam_ant*sy)/2,
                f"APD ≈ {apd_mm:.1f} mm", color="white",
                bbox=dict(facecolor="black", alpha=0.5), fontsize=10)

    ax.set_title(f"Axial QC — {level_row['level']}  (z={z_mid}, x={x_mid})")
    plt.tight_layout(); plt.show()

ok = meas_df[meas_df["severity"] >= 0]
if len(ok):
    qc_axial_level_v2(ct_arr2, ct_img2, seg_arr2, ok.iloc[0])



# === QC: draw axial APD column(s) for a measured level ===
def qc_axial_level(ct_arr, ct_img, seg_arr, level_row, x_band=X_BAND):
    z_mid, x_mid, up = int(level_row["z_mid"]), int(level_row["x_mid"]), int(level_row["upper"])
    sx, sy, sz = ct_img.GetSpacing()
    ax_ct  = ct_arr[z_mid].astype(np.float32)
    ax_seg = seg_arr[z_mid].astype(np.int32)

    body_box = _axial_body_edges(ax_seg, up)
    if body_box is None:
        print("No body mask on this slice.")
        return
    x_min, x_max, y_min, y_max, y_vb_post = body_box
    y_vb_ant = int(y_min)

    x0 = max(0, x_mid - x_band); x1 = min(ax_ct.shape[1]-1, x_mid + x_band)

    # compute lamina positions used
    y_lams = []
    for x in range(x0, x1+1):
        y_lam = _axial_lamina_anterior_y(ax_ct, y_vb_post, x)
        if y_lam is not None and y_lam > y_vb_post:
            y_lams.append(y_lam)
    y_lam_ant = int(np.median(y_lams)) if y_lams else None

    img = window_image(ax_ct, 400, 1800)
    ov  = overlay_segmentation(img, ax_seg)

    H,W = img.shape
    extent = [0, W*sx, H*sy, 0]
    fig, ax = plt.subplots(figsize=(6, 6*(H*sy)/(W*sx)))
    ax.imshow(ov, extent=extent); ax.set_aspect('equal'); ax.axis('off')

    # draw measurement column at x_mid
    x_mm = x_mid * sx
    ax.axvline(x_mm, color="white", lw=1.5, linestyle="--")

    # draw VB anterior/posterior and lamina lines along that column
    ax.hlines([y_vb_ant*sy, y_vb_post*sy], x_mm-4, x_mm+4, colors=["cyan","yellow"], lw=2)
    if y_lam_ant is not None:
        ax.hlines([y_lam_ant*sy], x_mm-4, x_mm+4, colors=["red"], lw=2)

        apd_mm = (y_lam_ant - y_vb_post) * sy
        ax.text(x_mm+6, (y_vb_post*sy + y_lam_ant*sy)/2,
                f"APD ≈ {apd_mm:.1f} mm", color="white",
                bbox=dict(facecolor="black", alpha=0.5), fontsize=10)

    ax.set_title(f"Axial QC — {level_row['level']}  (z={z_mid}, x={x_mid})")
    plt.tight_layout(); plt.show()

ok = meas_df[meas_df["severity"] >= 0]
if len(ok):
    qc_axial_level(ct_arr2, ct_img2, seg_arr2, ok.iloc[0])



# === Cell 6E: Axial APD/Torg with body-only mask + intensity lamina ===
import numpy as np
from scipy import ndimage as ndi

BONE_TH   = 250.0     # try 220–300 depending on kernel
X_BAND    = 5         # lateral median band around x_mid
Z_NEIGH   = 1         # allow slight z jitter
OPEN_SZ   = 7         # opening kernel to delete thin arches
KEEP_FRAC = 0.60      # keep anterior 60% of vertebra mask as "body"
RUN_LEN   = 3         # consecutive bone pixels to accept lamina

def _axial_body_only(ax_seg2d, vertebra_id):
    """Return (y_vb_ant, y_vb_post) for vertebral BODY only."""
    mask = (ax_seg2d == int(vertebra_id))
    if not mask.any(): 
        return None
    # remove thin posterior elements
    opened = ndi.binary_opening(mask, structure=np.ones((OPEN_SZ, OPEN_SZ)))
    m = opened if opened.any() else mask.copy()
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()
    # keep only anterior 60% to be body-only
    cut = int(y0 + KEEP_FRAC * (y1 - y0))
    body = m.copy(); body[cut+1:, :] = 0
    ys2 = np.where(body)[0]
    if ys2.size == 0: 
        return None
    return int(ys2.min()), int(ys2.max())

def _lamina_y_from_intensity(ax_ct2d, y_start_post, x_range):
    """Scan posteriorly for first robust bone run (intensity-based)."""
    H, W = ax_ct2d.shape
    y_min = max(0, y_start_post + 1)
    for y in range(y_min, H - RUN_LEN):
        # accept if ANY x in band has RUN_LEN consecutive bone pixels
        ok = False
        for x in x_range:
            col = ax_ct2d[y:y+RUN_LEN, int(x)]
            if np.all(col > BONE_TH):
                ok = True; break
        if ok:
            return y
    return None

def measure_level_APD_Torg_axial_v3(ct_arr, ct_img, seg_arr, up_id, z_mid, x_mid,
                                    x_band=X_BAND, z_neigh=Z_NEIGH):
    if z_mid is None or x_mid is None: 
        return None
    sx, sy, sz = ct_img.GetSpacing()
    z0 = max(0, int(z_mid) - z_neigh)
    z1 = min(ct_arr.shape[0]-1, int(z_mid) + z_neigh)
    x0 = max(0, int(x_mid) - x_band)
    x1 = min(ct_arr.shape[2]-1, int(x_mid) + x_band)

    apd_vals, torg_vals = [], []
    for z in range(z0, z1+1):
        ax_ct  = ct_arr[z].astype(np.float32)
        ax_seg = seg_arr[z].astype(np.int32)

        vb = _axial_body_only(ax_seg, up_id)
        if vb is None: 
            continue
        y_vb_ant, y_vb_post = vb

        y_lam = _lamina_y_from_intensity(ax_ct, y_vb_post, range(x0, x1+1))
        if y_lam is None or y_lam <= y_vb_post:
            continue

        apd_mm   = (y_lam - y_vb_post) * float(sy)
        vb_ap_mm = max(0.0, (y_vb_post - y_vb_ant) * float(sy))
        if vb_ap_mm <= 0:
            continue
        torg = apd_mm / vb_ap_mm
        apd_vals.append(apd_mm); torg_vals.append(torg)

    if not apd_vals:
        return None
    return {"APD_mm": float(np.median(apd_vals)),
            "Torg":   float(np.median(torg_vals)) if torg_vals else np.nan}

def severity_from_apd_torg(apd_mm, torg):
    # Severe <10 mm; Moderate 10–13 mm; Torg<0.8 up-bins near the boundary.
    if apd_mm < 10.0: s = 2
    elif apd_mm < 13.0: s = 1
    else: s = 0
    if not np.isnan(torg) and torg < 0.8 and 9.5 <= apd_mm <= 13.5:
        s = min(2, s+1)
    return s

# safe fallbacks (unchanged)
def safe_zmid(seg_arr, z_mid, upper_id, lower_id):
    if z_mid == z_mid: return int(round(z_mid))
    for vid in (int(upper_id), int(lower_id)):
        m = (seg_arr == vid)
        if m.any(): return int(np.median(np.argwhere(m)[:,0]))
    return seg_arr.shape[0] // 2

def safe_xmid(seg_arr, x_mid, upper_id):
    if x_mid == x_mid: return int(round(x_mid))
    m = (seg_arr == int(upper_id))
    if m.any():
        xs = np.argwhere(m)[:,2]
        return int(np.median(xs))
    return seg_arr.shape[2] // 2

# --- recompute
meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid = safe_zmid(seg_arr2, r["z_mid"], r["upper"], r["lower"])
    x_mid = safe_xmid(seg_arr2, r["x_mid"], r["upper"])
    up    = int(r["upper"])
    m = measure_level_APD_Torg_axial_v3(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid,
                          "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, **m, "severity": sev})
meas_df = pd.DataFrame(meas_rows)
print("[MEAS-v3] Body-only + intensity lamina:")
display(meas_df)



# QC: axial overlay with v3 logic
def qc_axial_level_v3(ct_arr, ct_img, seg_arr, level_row, x_band=X_BAND):
    z_mid, x_mid, up = int(level_row["z_mid"]), int(level_row["x_mid"]), int(level_row["upper"])
    sx, sy, sz = ct_img.GetSpacing()
    ax_ct  = ct_arr[z_mid].astype(np.float32)
    ax_seg = seg_arr[z_mid].astype(np.int32)

    vb = _axial_body_only(ax_seg, up)
    if vb is None: 
        print("No vertebral BODY on this slice."); return
    y_vb_ant, y_vb_post = vb
    x0 = max(0, x_mid - x_band); x1 = min(ax_ct.shape[1]-1, x_mid + x_band)
    y_lam = _lamina_y_from_intensity(ax_ct, y_vb_post, range(x0, x1+1))

    img = window_image(ax_ct, 400, 1800)
    ov  = overlay_segmentation(img, ax_seg)
    H,W = img.shape
    extent = [0, W*sx, H*sy, 0]
    fig, ax = plt.subplots(figsize=(6, 6*(H*sy)/(W*sx)))
    ax.imshow(ov, extent=extent); ax.set_aspect('equal'); ax.axis('off')
    x_mm = x_mid * sx
    ax.axvline(x_mm, color="white", lw=1.2, linestyle="--")
    ax.hlines([y_vb_ant*sy, y_vb_post*sy], x_mm-6, x_mm+6, colors=["cyan","yellow"], lw=2)
    if y_lam is not None:
        ax.hlines([y_lam*sy], x_mm-6, x_mm+6, colors=["red"], lw=2)
        apd_mm = (y_lam - y_vb_post) * sy
        ax.text(x_mm+6, (y_vb_post*sy + y_lam*sy)/2, f"APD ≈ {apd_mm:.1f} mm",
                color="white", bbox=dict(facecolor="black", alpha=0.5), fontsize=10)
    ax.set_title(f"Axial QC — {level_row['level']}  (z={z_mid}, x={x_mid})")
    plt.tight_layout(); plt.show()

ok = meas_df[meas_df["severity"] >= 0]
if len(ok):
    qc_axial_level_v3(ct_arr2, ct_img2, seg_arr2, ok.iloc[0])



# === Cell 6F: Axial APD/Torg with ROI crop + artifact cleanup + robust stats ===
import numpy as np
from scipy import ndimage as ndi

# Tunables (you can tweak)
BONE_TH     = 240.0      # HU threshold for intensity fallback (220–300)
OPEN_SZ     = 7          # opening kernel to delete thin arches from vertebra label
KEEP_FRAC   = 0.65       # keep anterior 65% of vertebra label as "body"
X_BAND      = 5          # measure across x in [x_mid ± X_BAND]
Z_NEIGH     = 1          # z ∈ [z_mid ± Z_NEIGH]
ROI_Y_FRONT = 6          # mm anterior margin in ROI (beyond y_vb_ant)
ROI_Y_BACK  = 25         # mm posterior margin in ROI (beyond y_vb_post)
ROI_X_MARGIN= 12         # mm lateral margin around x_mid
AREA_MIN    = 80         # px: remove small bone components (artifact spicules)
ROBUST_Q    = 0.75       # take 75th percentile APD across band (robust to outliers)
RUN_LEN     = 3          # consecutive bone px to accept lamina

def _axial_body_only(ax_seg2d, vertebra_id):
    m = (ax_seg2d == int(vertebra_id))
    if not m.any(): return None
    opened = ndi.binary_opening(m, structure=np.ones((OPEN_SZ, OPEN_SZ)))
    base = opened if opened.any() else m
    ys, xs = np.where(base)
    y0, y1 = ys.min(), ys.max()
    cut = int(y0 + KEEP_FRAC*(y1 - y0))
    body = base.copy(); body[cut+1:, :] = 0
    ys2 = np.where(body)[0]
    if ys2.size == 0: return None
    return int(ys2.min()), int(ys2.max())

def _px_from_mm(mm, spacing):  # helper
    return int(round(mm / float(spacing)))

def _clean_bone_union(ax_seg2d):
    # union all labels >0; remove tiny components
    bone = (ax_seg2d > 0).astype(np.uint8)
    lab, num = ndi.label(bone)
    if num == 0: return bone
    sizes = np.bincount(lab.ravel())
    kill = np.where(sizes < AREA_MIN)[0]
    bone[np.isin(lab, kill)] = 0
    return bone.astype(np.uint8)

def _lamina_y_from_union(ax_bone2d, y_start_post, x_range):
    # first robust bone posterior to body, requiring RUN_LEN consecutive bone px
    H, W = ax_bone2d.shape
    y_min = max(0, y_start_post + 1)
    for y in range(y_min, H-RUN_LEN):
        ok = False
        for x in x_range:
            if np.all(ax_bone2d[y:y+RUN_LEN, int(x)] > 0):
                ok = True; break
        if ok:
            return y
    return None

def measure_level_APD_Torg_axial_v4(ct_arr, ct_img, seg_arr, up_id, z_mid, x_mid):
    if z_mid is None or x_mid is None: return None
    sx, sy, sz = ct_img.GetSpacing()          # axial pixels: (sy, sx)
    z0 = max(0, int(z_mid) - Z_NEIGH)
    z1 = min(ct_arr.shape[0]-1, int(z_mid) + Z_NEIGH)

    # mm → px ROI margins
    dy_front = _px_from_mm(ROI_Y_FRONT, sy)
    dy_back  = _px_from_mm(ROI_Y_BACK,  sy)
    dxm      = _px_from_mm(ROI_X_MARGIN, sx)

    apd_all, torg_all = [], []

    for z in range(z0, z1+1):
        ax_ct  = ct_arr[z].astype(np.float32)
        ax_seg = seg_arr[z].astype(np.int32)

        # 1) BODY-only from label
        vb = _axial_body_only(ax_seg, up_id)
        if vb is None: 
            continue
        y_vb_ant, y_vb_post = vb

        # 2) ROI crop (“cone in”)
        H, W = ax_ct.shape
        y0 = max(0, y_vb_ant - dy_front)
        y1 = min(H-1, y_vb_post + dy_back)
        x0 = max(0, int(x_mid) - dxm)
        x1 = min(W-1, int(x_mid) + dxm)

        ax_ct_roi  = ax_ct[y0:y1+1, x0:x1+1]
        ax_seg_roi = ax_seg[y0:y1+1, x0:x1+1]
        y_vb_ant_r = y_vb_ant - y0
        y_vb_post_r= y_vb_post - y0
        x_mid_r    = int(x_mid) - x0

        # 3) bone union cleanup in ROI
        bone_u = _clean_bone_union(ax_seg_roi)  # 0/1

        # 4) measure across lateral band (robust to one bad column)
        xb0 = max(0, x_mid_r - X_BAND)
        xb1 = min(ax_ct_roi.shape[1]-1, x_mid_r + X_BAND)
        apds, torgs = [], []

        for x in range(xb0, xb1+1):
            y_lam = _lamina_y_from_union(bone_u, y_vb_post_r, [x])
            if y_lam is None or y_lam <= y_vb_post_r: 
                continue
            apd_mm   = (y_lam - y_vb_post_r) * float(sy)
            vb_ap_mm = max(0.0, (y_vb_post_r - y_vb_ant_r) * float(sy))
            if vb_ap_mm <= 0: 
                continue
            apds.append(apd_mm); torgs.append(apd_mm / vb_ap_mm)

        if len(apds) == 0:
            continue

        apd_rob = float(np.quantile(apds, ROBUST_Q))
        torg_rob= float(np.quantile(torgs, ROBUST_Q)) if len(torgs) else np.nan
        apd_all.append(apd_rob); torg_all.append(torg_rob)

    if len(apd_all) == 0:
        return None
    apd = float(np.median(apd_all))
    torg = float(np.median(torg_all)) if len(torg_all) else np.nan
    return {"APD_mm": apd, "Torg": torg}

def severity_from_apd_torg(apd_mm, torg):
    # Severe <10 mm; Moderate 10–13 mm; Torg<0.8 supports up-binning near boundary
    if apd_mm < 10.0: s = 2
    elif apd_mm < 13.0: s = 1
    else: s = 0
    if not np.isnan(torg) and torg < 0.8 and 9.5 <= apd_mm <= 13.5:
        s = min(2, s+1)
    return s

# Safe fallbacks (same as before)
def safe_zmid(seg_arr, z_mid, upper_id, lower_id):
    if z_mid == z_mid: return int(round(z_mid))
    for vid in (int(upper_id), int(lower_id)):
        m = (seg_arr == vid)
        if m.any(): return int(np.median(np.argwhere(m)[:,0]))
    return seg_arr.shape[0] // 2

def safe_xmid(seg_arr, x_mid, upper_id):
    if x_mid == x_mid: return int(round(x_mid))
    m = (seg_arr == int(upper_id))
    if m.any():
        xs = np.argwhere(m)[:,2]
        return int(np.median(xs))
    return seg_arr.shape[2] // 2

# --- recompute for this study with v4 ---
meas_rows = []
for r in perlevel_df.to_dict("records"):
    z_mid = safe_zmid(seg_arr2, r["z_mid"], r["upper"], r["lower"])
    x_mid = safe_xmid(seg_arr2, r["x_mid"], r["upper"])
    up    = int(r["upper"])
    m = measure_level_APD_Torg_axial_v4(ct_arr2, ct_img2, seg_arr2, up, z_mid, x_mid)
    if m is None:
        meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, "APD_mm": np.nan, "Torg": np.nan, "severity": -1})
        continue
    sev = severity_from_apd_torg(m["APD_mm"], m["Torg"])
    meas_rows.append({**r, "z_mid": z_mid, "x_mid": x_mid, **m, "severity": sev})

meas_df = pd.DataFrame(meas_rows)
print("[MEAS-v4] ROI + cleanup + robust APD:")
display(meas_df)



# === QC for v4: compact, aspect-correct panel ===
def qc_axial_level_v4(ct_arr, ct_img, seg_arr, level_row):
    z_mid, x_mid, up = int(level_row["z_mid"]), int(level_row["x_mid"]), int(level_row["upper"])
    sx, sy, sz = ct_img.GetSpacing()
    ax_ct  = ct_arr[z_mid].astype(np.float32)
    ax_seg = seg_arr[z_mid].astype(np.int32)

    vb = _axial_body_only(ax_seg, up)
    if vb is None: 
        print("No vertebral BODY on this slice."); return
    y_vb_ant, y_vb_post = vb

    # ROI as above
    dy_front = _px_from_mm(ROI_Y_FRONT, sy); dy_back = _px_from_mm(ROI_Y_BACK, sy); dxm = _px_from_mm(ROI_X_MARGIN, sx)
    H,W = ax_ct.shape
    y0 = max(0, y_vb_ant - dy_front); y1 = min(H-1, y_vb_post + dy_back)
    x0 = max(0, int(x_mid) - dxm);    x1 = min(W-1, int(x_mid) + dxm)
    ct_roi  = ax_ct[y0:y1+1, x0:x1+1]
    seg_roi = ax_seg[y0:y1+1, x0:x1+1]
    bone_u  = _clean_bone_union(seg_roi)

    img = window_image(ct_roi, center=500, width=2200)  # boney display
    ov  = overlay_segmentation(img, seg_roi)

    # measure at center column after cleanup (for display)
    y_vb_post_r = y_vb_post - y0
    x_mid_r     = int(x_mid) - x0
    y_lam = _lamina_y_from_union(bone_u, y_vb_post_r, [x_mid_r])

    H2,W2 = img.shape
    extent = [0, W2*sx, H2*sy, 0]
    fig, ax = plt.subplots(figsize=(5.8, 5.8*(H2*sy)/(W2*sx)))
    ax.imshow(ov, extent=extent); ax.set_aspect('equal'); ax.axis('off')
    ax.axvline(x_mid_r*sx, color="white", lw=1.2, linestyle="--")
    ax.hlines([ (y_vb_post_r)*sy ], x_mid_r*sx-6, x_mid_r*sx+6, colors="yellow", lw=2, label="VB posterior")
    if y_lam is not None:
        ax.hlines([ y_lam*sy ], x_mid_r*sx-6, x_mid_r*sx+6, colors="red", lw=2, label="Lamina anterior")
        apd_mm = (y_lam - y_vb_post_r) * sy
        ax.text(x_mid_r*sx+6, (y_vb_post_r*sy + y_lam*sy)/2,
                f"APD ≈ {apd_mm:.1f} mm", color="white",
                bbox=dict(facecolor="black", alpha=0.55), fontsize=10)
    ax.set_title(f"Axial QC — {level_row['level']} (coned-in)")
    plt.tight_layout(); plt.show()

ok = meas_df[meas_df["severity"] >= 0]
if len(ok):
    qc_axial_level_v4(ct_arr2, ct_img2, seg_arr2, ok.iloc[0])



# === Cell 6G: Midline-refined axial APD/Torg + dual-view QC ===
import numpy as np
from scipy import ndimage as ndi

# Tunables
BONE_TH     = 240.0
OPEN_SZ     = 7
KEEP_FRAC   = 0.65         # keep anterior 65% of vertebra mask as "body-only"
X_BAND      = 5            # band around the midline for robust stats
Z_NEIGH     = 1            # z jitter around disc mid-plane
ROI_Y_FRONT = 10           # mm anterior margin in ROI (beyond VB anterior)
ROI_Y_BACK  = 30           # mm posterior margin in ROI (beyond VB posterior)
ROI_X_MARG  = 20           # mm lateral margin around x_mid
AREA_MIN    = 120          # px: remove tiny bone components in ROI
RUN_LEN     = 3            # consecutive bone px (de-streaking)
ROBUST_Q    = 0.75         # percentile across band (robust APD)
MED_FILT    = (3, 1)       # median filter kernel for ROI (reduce streaks)

def _px_from_mm(mm, sp): return int(round(mm / float(sp)))

def _body_only_mask(ax_seg2d, vid):
    m = (ax_seg2d == int(vid))
    if not m.any(): return None
    opened = ndi.binary_opening(m, structure=np.ones((OPEN_SZ, OPEN_SZ)))
    base = opened if opened.any() else m
    ys, xs = np.where(base)
    y0, y1 = ys.min(), ys.max()
    cut = int(y0 + KEEP_FRAC * (y1 - y0))
    body = base.copy()
    body[cut+1:, :] = 0
    return body if body.any() else None

def _refine_x_mid_from_bodies(ax_seg2d, up_id, lo_id, x_default):
    xs = []
    for vid in (int(up_id), int(lo_id)):
        m = (ax_seg2d == vid)
        if m.any():
            xs.append(int(np.median(np.where(m)[1])))
    return int(np.mean(xs)) if xs else int(x_default)

def _clean_bone_union(ax_seg2d):
    bone = (ax_seg2d > 0).astype(np.uint8)
    lab, num = ndi.label(bone)
    if num == 0: return bone
    sizes = np.bincount(lab.ravel())
    kill = np.where(sizes < AREA_MIN)[0]
    bone[np.isin(lab, kill)] = 0
    return bone.astype(np.uint8)

def _first_bone_run_y(bone_bin2d, y_from, x):
    H, W = bone_bin2d.shape
    y0 = max(0, int(y_from) + 1)
    for y in range(y0, H - RUN_LEN):
        if np.all(bone_bin2d[y:y+RUN_LEN, int(x)] > 0):
            return y
    return None

def _safe_zmid(seg_arr, z_mid, up_id, lo_id):
    if z_mid == z_mid: return int(round(z_mid))
    for vid in (int(up_id), int(lo_id)):
        m = (seg_arr == vid)
        if m.any(): return int(np.median(np.argwhere(m)[:,0]))
    return seg_arr.shape[0] // 2

def _safe_xmid(seg_arr, x_mid, up_id, lo_id, z_mid):
    if x_mid == x_mid: return int(round(x_mid))
    # try centroid at z_mid from both vertebrae; else default to image center
    if 0 <= z_mid < seg_arr.shape[0]:
        return _refine_x_mid_from_bodies(seg_arr[z_mid], up_id, lo_id, seg_arr.shape[2]//2)
    return seg_arr.shape[2] // 2

def measure_axial_midline(ct_arr, ct_img, seg_arr, up_id, lo_id, z_mid, x_mid):
    """Returns dict with APD_mid (mm), APD_band (mm), Torg_mid, Torg_band, and plotting info."""
    sx, sy, sz = ct_img.GetSpacing()
    z_mid = _safe_zmid(seg_arr, z_mid, up_id, lo_id)
    x_mid = _safe_xmid(seg_arr, x_mid, up_id, lo_id, z_mid)

    z0 = max(0, z_mid - Z_NEIGH)
    z1 = min(ct_arr.shape[0]-1, z_mid + Z_NEIGH)

    # compute on each nearby z, collect stats
    apd_mid_list, torg_mid_list, apd_band_list, torg_band_list = [], [], [], []
    # store last ROI slices for QC
    qc_payload = None

    for z in range(z0, z1+1):
        ax_ct  = ct_arr[z].astype(np.float32)
        ax_seg = seg_arr[z].astype(np.int32)

        body_mask = _body_only_mask(ax_seg, up_id)
        if body_mask is None: 
            continue

        ys, xs = np.where(body_mask)
        y_vb_ant = int(ys.min()); y_vb_post = int(ys.max())

        # ROI (cone-in, but wide enough to include entire spine)
        dy_f = _px_from_mm(ROI_Y_FRONT, sy)
        dy_b = _px_from_mm(ROI_Y_BACK,  sy)
        dxm  = _px_from_mm(ROI_X_MARG,  sx)

        H,W = ax_ct.shape
        y0 = max(0, y_vb_ant - dy_f); y1 = min(H-1, y_vb_post + dy_b)
        x0 = max(0, x_mid - dxm);     x1 = min(W-1, x_mid + dxm)

        ct_roi  = ax_ct[y0:y1+1, x0:x1+1]
        seg_roi = ax_seg[y0:y1+1, x0:x1+1]
        body_roi= body_mask[y0:y1+1, x0:x1+1]
        bone_u  = _clean_bone_union(seg_roi)

        # slight de-streak
        ct_roi_f = ndi.median_filter(ct_roi, size=MED_FILT)

        # midline & band in ROI coords
        x_mid_r  = int(x_mid) - x0
        y_vb_ant_r, y_vb_post_r = y_vb_ant - y0, y_vb_post - y0
        xb0 = max(0, x_mid_r - X_BAND); xb1 = min(ct_roi.shape[1]-1, x_mid_r + X_BAND)

        # --- MIDLINE measurement
        y_lam_mid = _first_bone_run_y(bone_u, y_vb_post_r, x_mid_r)
        if y_lam_mid is not None and y_lam_mid > y_vb_post_r:
            apd_mid = (y_lam_mid - y_vb_post_r) * float(sy)
            vb_ap   = max(0.0, (y_vb_post_r - y_vb_ant_r) * float(sy))
            if vb_ap > 0:
                apd_mid_list.append(apd_mid)
                torg_mid_list.append(apd_mid / vb_ap)

        # --- BAND-ROBUST measurement
        apds, torgs = [], []
        for x in range(xb0, xb1+1):
            y_lam = _first_bone_run_y(bone_u, y_vb_post_r, x)
            if y_lam is None or y_lam <= y_vb_post_r: 
                continue
            apd = (y_lam - y_vb_post_r) * float(sy)
            vb_ap = max(0.0, (y_vb_post_r - y_vb_ant_r) * float(sy))
            if vb_ap > 0:
                apds.append(apd); torgs.append(apd / vb_ap)
        if apds:
            apd_band_list.append(float(np.quantile(apds, ROBUST_Q)))
            torg_band_list.append(float(np.quantile(torgs, ROBUST_Q)))

        qc_payload = dict(z=z, y0=y0, y1=y1, x0=x0, x1=x1, 
                          x_mid_r=x_mid_r, y_vb_ant_r=y_vb_ant_r, y_vb_post_r=y_vb_post_r,
                          y_lam_mid=y_lam_mid, ct_roi=ct_roi_f, seg_roi=seg_roi)

    if not (apd_mid_list or apd_band_list):
        return None

    res = dict(
        APD_mid_mm = np.median(apd_mid_list) if apd_mid_list else np.nan,
        APD_band_mm= np.median(apd_band_list) if apd_band_list else np.nan,
        Torg_mid   = np.median(torg_mid_list) if torg_mid_list else np.nan,
        Torg_band  = np.median(torg_band_list) if torg_band_list else np.nan,
        z_mid_used = z_mid,
        x_mid_used = x_mid,
        qc=qc_payload
    )
    return res

def severity_from_apd(apd_mm):
    # severe <10, moderate 10–13, else none/mild
    if np.isnan(apd_mm): return -1
    return 2 if apd_mm < 10.0 else (1 if apd_mm < 13.0 else 0)

# ---- run over this study
rows = []
for r in perlevel_df.to_dict("records"):
    m = measure_axial_midline(ct_arr2, ct_img2, seg_arr2, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
    if m is None:
        rows.append({**r, "APD_mid_mm": np.nan, "APD_band_mm": np.nan, "Torg_mid": np.nan, "Torg_band": np.nan,
                     "severity_mid": -1, "severity_band": -1})
    else:
        rows.append({**r, **{k:v for k,v in m.items() if k!="qc"},
                     "severity_mid": severity_from_apd(m["APD_mid_mm"]),
                     "severity_band": severity_from_apd(m["APD_band_mm"])})
meas_df = pd.DataFrame(rows)
print("[MEAS-midline] APD (midline & band) and severities:")
display(meas_df)

# ---- Dual-view QC (full + ROI) for the first measurable level
def qc_dual_full_and_roi(ct_arr, ct_img, seg_arr, level_row, qc_payload,
                         window_full=(400,1800), window_roi=(500,2200)):
    sx, sy, sz = ct_img.GetSpacing()
    z = qc_payload["z"]; y0,y1,x0,x1 = qc_payload["y0"],qc_payload["y1"],qc_payload["x0"],qc_payload["x1"]
    x_mid_r = qc_payload["x_mid_r"]; y_vb_ant_r=qc_payload["y_vb_ant_r"]; y_vb_post_r=qc_payload["y_vb_post_r"]
    y_lam_mid = qc_payload["y_lam_mid"]; ct_roi = qc_payload["ct_roi"]; seg_roi = qc_payload["seg_roi"]

    # full-FOV left
    full = window_image(ct_arr[z], *window_full); seg_full = seg_arr[z]
    fig, axs = plt.subplots(1,2, figsize=(11,5.5))

    extent_full = [0, full.shape[1]*sx, full.shape[0]*sy, 0]
    axs[0].imshow(overlay_segmentation(full, seg_full), extent=extent_full); axs[0].set_aspect('equal'); axs[0].axis('off')
    axs[0].axvline(meas_df.loc[0,"x_mid"], color="white", lw=1.2, linestyle="--")
    axs[0].set_title(f"Full axial (z={z})")

    # ROI right
    roi = window_image(ct_roi, *window_roi)
    extent_roi = [0, roi.shape[1]*sx, roi.shape[0]*sy, 0]
    axs[1].imshow(overlay_segmentation(roi, seg_roi), extent=extent_roi); axs[1].set_aspect('equal'); axs[1].axis('off')
    x_mm = x_mid_r * sx
    axs[1].axvline(x_mm, color="white", lw=1.2, linestyle="--")
    axs[1].hlines([y_vb_ant_r*sy, y_vb_post_r*sy], x_mm-8, x_mm+8, colors=["cyan","yellow"], lw=2)
    if y_lam_mid is not None:
        axs[1].hlines([y_lam_mid*sy], x_mm-8, x_mm+8, colors=["red"], lw=2)
        apd = (y_lam_mid - y_vb_post_r) * sy
        axs[1].text(x_mm+6, (y_vb_post_r*sy + y_lam_mid*sy)/2,
                    f"APD(mid) ≈ {apd:.1f} mm", color="white",
                    bbox=dict(facecolor="black", alpha=0.55), fontsize=10)
    axs[1].set_title(f"Coned-in ROI")

    plt.suptitle(f"QC — {level_row['level']} (midline & ROI)", y=0.98, fontsize=12)
    plt.tight_layout(); plt.show()

# pick first measurable level
idx = meas_df.index[meas_df["APD_mid_mm"].notna()].tolist()
if idx:
    # we need the qc payload from a fresh measurement call (to carry arrays)
    r = perlevel_df.iloc[idx[0]].to_dict()
    m = measure_axial_midline(ct_arr2, ct_img2, seg_arr2, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
    if m and m["qc"]:
        qc_dual_full_and_roi(ct_arr2, ct_img2, seg_arr2, r, m["qc"])



# === Cell 6H: Robust axial APD/Torg (label∪HU bone, non-bone run) ===
import numpy as np
from scipy import ndimage as ndi

# Tunables (adjust if needed)
HU_BONE_TH   = 240.0   # HU threshold to add to bone union (try 220–300 for soft/bone kernels)
OPEN_SZ       = 7      # opening to remove thin posterior elements from vertebra mask
KEEP_FRAC     = 0.65   # keep anterior 65% of vertebra mask as body-only
Z_NEIGH       = 1      # z ∈ [z_mid ± Z_NEIGH]
X_BAND        = 5      # measure also across ±X_BAND cols around midline (robust)
AREA_MIN      = 120    # px: drop tiny bone comps (spicules)
RUN_LEN       = 3      # consecutive bone px to accept "true" lamina (de-streak)
ROBUST_Q      = 0.75   # percentile for band (robust) APD
MED_FILT      = (3,1)  # median filter on ROI to reduce streaks (H×W)

def _px_from_mm(mm, sp): return int(round(mm / float(sp)))

def _body_only_mask(ax_seg2d, vid):
    m = (ax_seg2d == int(vid))
    if not m.any(): return None
    opened = ndi.binary_opening(m, structure=np.ones((OPEN_SZ, OPEN_SZ)))
    base = opened if opened.any() else m
    ys = np.where(base)[0]
    y0, y1 = ys.min(), ys.max()
    cut = int(y0 + KEEP_FRAC*(y1-y0))
    body = base.copy(); body[cut+1:, :] = 0
    return body if body.any() else None

def _refine_midline_x(ax_seg2d, up_id, lo_id, x_default):
    xs=[]
    for vid in (int(up_id), int(lo_id)):
        m=(ax_seg2d==vid)
        if m.any(): xs.append(int(np.median(np.where(m)[1])))
    return int(np.mean(xs)) if xs else int(x_default)

def _bone_union(ax_seg2d, ax_ct2d):
    bone = (ax_seg2d>0) | (ax_ct2d>HU_BONE_TH)
    # remove tiny comps
    lab, num = ndi.label(bone.astype(np.uint8))
    if num:
        sizes=np.bincount(lab.ravel())
        bone[np.isin(lab, np.where(sizes<AREA_MIN)[0])]=0
    # small closing to bridge minor gaps in lamina
    bone = ndi.binary_closing(bone, structure=np.ones((3,3)))
    return bone.astype(np.uint8)

def _first_nonbone_run_len(nonbone, y_from, x):
    """Length in px of continuous non-bone after y_from along +Y."""
    H = nonbone.shape[0]
    y0 = max(0, y_from+1)
    cnt=0
    for y in range(y0, H):
        if nonbone[y, int(x)]: cnt += 1
        else: break
    return cnt

def _safe_z(seg, z_mid, up, lo):
    if z_mid==z_mid: return int(round(z_mid))
    for vid in (int(up), int(lo)):
        m=(seg==vid)
        if m.any(): return int(np.median(np.argwhere(m)[:,0]))
    return seg.shape[0]//2

def _safe_x(seg, x_mid, up, lo, z_mid):
    if x_mid==x_mid: return int(round(x_mid))
    if 0<=z_mid<seg.shape[0]:
        return _refine_midline_x(seg[z_mid], up, lo, seg.shape[2]//2)
    return seg.shape[2]//2

def measure_axial_robust(ct_arr, ct_img, seg_arr, up_id, lo_id, z_mid, x_mid,
                         roi_front_mm=10, roi_back_mm=30, roi_lat_mm=20):
    """Robust APD on disc midline & band using non-bone run length."""
    sx, sy, sz = ct_img.GetSpacing()
    z_mid = _safe_z(seg_arr, z_mid, up_id, lo_id)
    x_mid = _safe_x(seg_arr, x_mid, up_id, lo_id, z_mid)

    z0=max(0,z_mid-Z_NEIGH); z1=min(ct_arr.shape[0]-1,z_mid+Z_NEIGH)
    apd_mid_list, apd_band_list, torg_mid_list, torg_band_list = [], [], [], []
    qc_payload=None

    for z in range(z0,z1+1):
        ax_ct = ct_arr[z].astype(np.float32)
        ax_seg= seg_arr[z].astype(np.int32)

        body = _body_only_mask(ax_seg, up_id)
        if body is None: continue
        ys = np.where(body)[0]
        y_ant, y_post = int(ys.min()), int(ys.max())

        # ROI
        dyf=_px_from_mm(roi_front_mm, sy); dyb=_px_from_mm(roi_back_mm, sy); dx=_px_from_mm(roi_lat_mm, sx)
        H,W=ax_ct.shape
        y0=max(0, y_ant-dyf); y1=min(H-1, y_post+dyb)
        x0=max(0, x_mid-dx);  x1=min(W-1, x_mid+dx)

        ct_roi  = ndi.median_filter(ax_ct[y0:y1+1, x0:x1+1], size=MED_FILT)
        seg_roi = ax_seg[y0:y1+1, x0:x1+1]
        body_r  = body[y0:y1+1, x0:x1+1]
        bone    = _bone_union(seg_roi, ct_roi)
        nonbone = (~bone.astype(bool)).astype(np.uint8)

        x_mid_r = int(x_mid)-x0
        y_ant_r, y_post_r = y_ant-y0, y_post-y0

        # MIDLINE
        gap_px = _first_nonbone_run_len(nonbone, y_post_r, x_mid_r)
        if gap_px>0:
            apd_mid = gap_px*sy
            vb_ap   = max(0.0, (y_post_r - y_ant_r)*sy)
            if vb_ap>0:
                apd_mid_list.append(apd_mid); torg_mid_list.append(apd_mid/vb_ap)

        # BAND (robust)
        xb0=max(0, x_mid_r-X_BAND); xb1=min(ct_roi.shape[1]-1, x_mid_r+X_BAND)
        apds=[]; torgs=[]
        for x in range(xb0, xb1+1):
            gpx=_first_nonbone_run_len(nonbone, y_post_r, x)
            if gpx<=0: continue
            apd=gpx*sy; vb_ap=max(0.0, (y_post_r-y_ant_r)*sy)
            if vb_ap>0: apds.append(apd); torgs.append(apd/vb_ap)
        if apds:
            apd_band_list.append(float(np.quantile(apds, ROBUST_Q)))
            torg_band_list.append(float(np.quantile(torgs, ROBUST_Q)))

        qc_payload=dict(z=z, y0=y0, y1=y1, x0=x0, x1=x1, x_mid_r=x_mid_r,
                        y_ant_r=y_ant_r, y_post_r=y_post_r, ct_roi=ct_roi,
                        seg_roi=seg_roi, nonbone=nonbone)

    if not (apd_mid_list or apd_band_list): return None
    return {
        "APD_mid_mm":  np.median(apd_mid_list)  if apd_mid_list  else np.nan,
        "APD_band_mm": np.median(apd_band_list) if apd_band_list else np.nan,
        "Torg_mid":    np.median(torg_mid_list) if torg_mid_list else np.nan,
        "Torg_band":   np.median(torg_band_list)if torg_band_list else np.nan,
        "z_mid_used": z_mid, "x_mid_used": x_mid, "qc": qc_payload
    }

def sev_from_apd(a): 
    if np.isnan(a): return -1
    return 2 if a<10 else (1 if a<13 else 0)

# ---- recompute on current study using robust method
rows=[]
for r in perlevel_df.to_dict("records"):
    m = measure_axial_robust(ct_arr2, ct_img2, seg_arr2, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
    if m is None:
        rows.append({**r, "APD_mid_mm":np.nan, "APD_band_mm":np.nan, "Torg_mid":np.nan, "Torg_band":np.nan,
                     "severity_mid":-1, "severity_band":-1})
    else:
        rows.append({**r, **{k:v for k,v in m.items() if k!="qc"},
                     "severity_mid":  sev_from_apd(m["APD_mid_mm"]),
                     "severity_band": sev_from_apd(m["APD_band_mm"])})
meas_df = pd.DataFrame(rows)
print("[MEAS-robust] APD (midline & band) and severities:")
display(meas_df)

# ---- dual-view QC (full + ROI) reusing 6G but with robust non-bone overlay
def qc_dual_full_and_roi_robust(ct_arr, ct_img, seg_arr, level_row, qc, window_full=(400,1800), window_roi=(500,2200)):
    sx, sy, sz = ct_img.GetSpacing()
    z=qc["z"]; y0,y1,x0,x1 = qc["y0"],qc["y1"],qc["x0"],qc["x1"]
    x_mid_r=qc["x_mid_r"]; y_ant_r=qc["y_ant_r"]; y_post_r=qc["y_post_r"]
    ct_roi=qc["ct_roi"]; seg_roi=qc["seg_roi"]; nonbone=qc["nonbone"]

    # full view
    full=window_image(ct_arr[z], *window_full); seg_full=seg_arr[z]
    fig,axs=plt.subplots(1,2,figsize=(11,5.5))
    extent_full=[0, full.shape[1]*sx, full.shape[0]*sy, 0]
    axs[0].imshow(overlay_segmentation(full, seg_full), extent=extent_full); axs[0].axis('off'); axs[0].set_aspect('equal')
    axs[0].axvline((x0+x_mid_r)*sx, color='white', lw=1.2, ls='--')
    axs[0].set_title(f"Full axial (z={z})")

    # ROI
    roi=window_image(ct_roi, *window_roi)
    extent_roi=[0, roi.shape[1]*sx, roi.shape[0]*sy, 0]
    axs[1].imshow(overlay_segmentation(roi, seg_roi), extent=extent_roi); axs[1].axis('off'); axs[1].set_aspect('equal')
    # overlay non-bone (canal) as faint green
    canal = np.ma.masked_where(nonbone==0, nonbone)
    axs[1].imshow(canal, extent=extent_roi, cmap='Greens', alpha=0.25, interpolation='nearest')
    x_mm=x_mid_r*sx
    axs[1].axvline(x_mm, color='white', lw=1.2, ls='--')
    axs[1].hlines([y_ant_r*sy, y_post_r*sy], x_mm-8, x_mm+8, colors=['cyan','yellow'], lw=2)
    # compute midline gap for display
    gap_px=_first_nonbone_run_len(nonbone, y_post_r, x_mid_r)
    if gap_px>0:
        apd=gap_px*sy
        axs[1].hlines([(y_post_r+gap_px)*sy], x_mm-8, x_mm+8, colors='red', lw=2)
        axs[1].text(x_mm+6, (y_post_r*sy + (y_post_r+gap_px)*sy)/2,
                    f"APD(mid) ≈ {apd:.1f} mm", color='white',
                    bbox=dict(fc='black', alpha=0.55), fontsize=10)
    axs[1].set_title("Coned-in ROI (robust canal)")

    plt.suptitle(f"QC — {level_row['level']} (midline & ROI)", y=0.98, fontsize=12)
    plt.tight_layout(); plt.show()

# Show QC for first measurable level
idx = meas_df.index[meas_df["APD_mid_mm"].notna()].tolist()
if idx:
    r = perlevel_df.iloc[idx[0]].to_dict()
    m = measure_axial_robust(ct_arr2, ct_img2, seg_arr2, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
    if m and m["qc"]:
        qc_dual_full_and_roi_robust(ct_arr2, ct_img2, seg_arr2, r, m["qc"])



# === Cell 7H: Batch runner for all eligible studies with cervical segs ===
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Build "eligible" list once (has seg and at least one C1..C7 voxel)
eligible = []
for p in study_dirs:
    if p.name not in seg_map: 
        continue
    img, arr = load_seg_nifti(seg_map[p.name])
    if np.any(np.isin(arr, np.arange(1,8))):
        eligible.append(p)
print(f"[BATCH] Eligible cervical studies: {len(eligible)}")

# Process N studies (set N=None for all)
N = None
rows_all = []
for p in tqdm(eligible[:N], desc="Measuring APD"):
    # load best CT series
    ct_img, ct_arr, files, uid = load_best_series_any(p)
    if ct_img is None: 
        continue
    seg_img, seg_arr = load_seg_nifti(seg_map[p.name])
    # resample seg→CT if needed
    same_geom = (ct_img.GetSize()==seg_img.GetSize() and
                 np.allclose(ct_img.GetSpacing(), seg_img.GetSpacing(), atol=1e-3) and
                 tuple(ct_img.GetDirection())==tuple(seg_img.GetDirection()))
    if not same_geom:
        seg_arr = sitk.GetArrayFromImage(resample_like(seg_img, ct_img, True))

    # per-level planes (C2/3..C6/7)
    recs=[]
    for lid in range(2,8):
        upper, lower = lid, min(lid+1,7)
        # disc mid-z
        up = (seg_arr==upper); lo=(seg_arr==lower)
        z_mid = int(np.median(np.argwhere(up)[:,0])) if up.any() else (int(np.median(np.argwhere(lo)[:,0])) if lo.any() else seg_arr.shape[0]//2)
        # initial x at upper body center; refined inside measure_axial_robust
        x_mid = int(np.median(np.where(up)[1])) if up.any() else seg_arr.shape[2]//2
        recs.append({"study_id": p.name, "level": f"C{upper}/C{lower}", "upper":upper, "lower":lower,
                     "z_mid": z_mid, "x_mid": x_mid})
    perlevel_df_batch = pd.DataFrame(recs)

    # robust measurements
    for r in perlevel_df_batch.to_dict("records"):
        m = measure_axial_robust(ct_arr, ct_img, seg_arr, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
        if m is None:
            rows_all.append({**r, "APD_mid_mm":np.nan, "APD_band_mm":np.nan, "Torg_mid":np.nan, "Torg_band":np.nan,
                             "severity_mid":-1, "severity_band":-1})
        else:
            rows_all.append({**r, **{k:v for k,v in m.items() if k!="qc"},
                             "severity_mid": sev_from_apd(m["APD_mid_mm"]),
                             "severity_band": sev_from_apd(m["APD_band_mm"])})
# Save cohort CSV
cohort_df = pd.DataFrame(rows_all)
csv_out = "/kaggle/working/ccs_cohort_apd.csv"
cohort_df.to_csv(csv_out, index=False)
print(f"[SAVE] Cohort measurements -> {csv_out}\nshape={cohort_df.shape}")

# Quick plots (overall)
ok_mid = cohort_df["APD_mid_mm"].dropna()
ok_band = cohort_df["APD_band_mm"].dropna()
plt.figure(figsize=(7,3)); plt.hist(ok_band, bins=30, alpha=0.8)
plt.axvline(10, color="r", ls="--", label="Severe < 10 mm"); plt.axvline(13, color="orange", ls="--", label="Moderate 10–13 mm")
plt.xlabel("AP canal diameter (mm), band-robust"); plt.ylabel("Count"); plt.legend(); plt.title("Cervical APD (band-robust) — cohort")
plt.tight_layout(); plt.show()

# Per-level medians
med = cohort_df.groupby("level")["APD_band_mm"].median().reset_index()
print("[SUMMARY] Median APD_band_mm by level:"); display(med)



# === Final: fast batch + report (no training) ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ---------- knobs ----------
FAST_N = 10                     # how many eligible studies to process quickly
LEVEL_WHITELIST = {"C4/C5","C5/C6","C6/C7"}  # focus cohort metrics on these
QC_EXAMPLES = 3                 # number of dual-view QC panels to render
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- safety checks ----------
assert 'study_dirs' in globals() and 'seg_map' in globals(), "Run the EDA cells first."
assert 'load_seg_nifti' in globals() and 'load_best_series_any' in globals(), "Run the loader cells."
assert 'resample_like' in globals(), "Missing resample_like."
assert 'measure_axial_robust' in globals(), "Run the robust measurement cell (6H)."

# ---------- discover eligible (has segmentation and some C1..C7 voxels) ----------
eligible = []
for p in study_dirs:
    if p.name not in seg_map:
        continue
    _, seg_arr0 = load_seg_nifti(seg_map[p.name])
    if np.any(np.isin(seg_arr0, np.arange(1,8))):
        eligible.append(p)
print(f"[BATCH] Eligible cervical studies with segs: {len(eligible)}")

# ---------- process a fast subset ----------
rows_all = []
use_list = eligible[:FAST_N] if FAST_N is not None else eligible
print(f"[RUN] Measuring {len(use_list)} studies...")

for p in tqdm(use_list, desc="APD measure"):
    # 1) CT series
    ct_img, ct_arr, files, uid = load_best_series_any(p)
    if ct_img is None:
        continue
    # 2) segmentation (NIfTI)
    seg_img, seg_arr = load_seg_nifti(seg_map[p.name])
    # 3) resample seg -> CT geometry if needed
    same_geom = (ct_img.GetSize()==seg_img.GetSize() and
                 np.allclose(ct_img.GetSpacing(), seg_img.GetSpacing(), atol=1e-3) and
                 tuple(ct_img.GetDirection())==tuple(seg_img.GetDirection()))
    if not same_geom:
        seg_arr = sitk.GetArrayFromImage(resample_like(seg_img, ct_img, True))

    # 4) define disc planes C2/3..C6/7 (simple, robust)
    perlevel = []
    for lid in range(2,8):  # C2..C7 => planes C2/3..C7/7 (C7/7 ignored later)
        upper, lower = lid, min(lid+1,7)
        up = (seg_arr==upper); lo=(seg_arr==lower)
        if up.any():
            z_mid = int(np.median(np.argwhere(up)[:,0]))
            x_mid = int(np.median(np.where(up)[1]))
        elif lo.any():
            z_mid = int(np.median(np.argwhere(lo)[:,0]))
            x_mid = int(np.median(np.where(lo)[1]))
        else:
            z_mid = seg_arr.shape[0]//2
            x_mid = seg_arr.shape[2]//2
        perlevel.append({"study_id": p.name, "level": f"C{upper}/C{lower}",
                         "upper":upper, "lower":lower, "z_mid":z_mid, "x_mid":x_mid})
    perlevel_df_batch = pd.DataFrame(perlevel)

    # 5) robust measurements
    for r in perlevel_df_batch.to_dict("records"):
        m = measure_axial_robust(ct_arr, ct_img, seg_arr, r["upper"], r["lower"], r["z_mid"], r["x_mid"])
        if m is None:
            rows_all.append({**r, "APD_mid_mm":np.nan, "APD_band_mm":np.nan, "Torg_mid":np.nan, "Torg_band":np.nan})
        else:
            rows_all.append({**r, **{k:v for k,v in m.items() if k!="qc"}})

# ---------- tidy & final severity ----------
def sev_from_apd(a):
    if np.isnan(a): return -1
    return 2 if a < 10.0 else (1 if a < 13.0 else 0)

cohort_df = pd.DataFrame(rows_all)
# choose band-robust when available, else midline
cohort_df["APD_final_mm"] = np.where(cohort_df["APD_band_mm"].notna(),
                                     cohort_df["APD_band_mm"], cohort_df["APD_mid_mm"])
cohort_df["severity_final"] = cohort_df["APD_final_mm"].apply(sev_from_apd)

csv_out = "/kaggle/working/ccs_cohort_apd_final.csv"
cohort_df.to_csv(csv_out, index=False)
print(f"[SAVE] Cohort CSV -> {csv_out}  shape={cohort_df.shape}")

# ---------- headline summary (focus on C4–C7) ----------
cohort_eval = cohort_df[cohort_df["level"].isin(LEVEL_WHITELIST)].copy()
n_levels = cohort_eval["APD_final_mm"].notna().sum()
n_studies = cohort_eval["study_id"].nunique()
triage_flag = (cohort_eval
               .assign(flag=lambda d: d["severity_final"].ge(1))
               .groupby("study_id")["flag"].any()
               .mean())
print(f"[SUMMARY] Levels with measurements (C4–C7): {n_levels} across {n_studies} studies")
print(f"[SUMMARY] % patients flagged (moderate+severe at any C4–C7): {100*triage_flag:.1f}%")

medians = (cohort_eval
           .groupby("level")["APD_final_mm"]
           .median()
           .reindex(sorted(LEVEL_WHITELIST)))
print("[SUMMARY] Per-level median APD_final_mm (mm):")
print(medians)

# ---------- quick plots ----------
ok = cohort_eval["APD_final_mm"].dropna()
plt.figure(figsize=(7,3.3))
plt.hist(ok, bins=28, alpha=0.85)
plt.axvline(10, color="r", ls="--", label="Severe < 10 mm")
plt.axvline(13, color="orange", ls="--", label="Moderate 10–13 mm")
plt.xlabel("AP canal diameter (mm) — APD_final"); plt.ylabel("Count"); plt.legend()
plt.title("Cervical APD (C4–C7) — cohort")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3.3))
lvls = sorted(list(LEVEL_WHITELIST))
data = [cohort_eval.loc[cohort_eval["level"]==L, "APD_final_mm"].dropna().values for L in lvls]
plt.boxplot(data, labels=lvls, showmeans=True)
plt.ylabel("AP canal diameter (mm)"); plt.title("Per-level APD (C4–C7)")
plt.tight_layout(); plt.show()

# ---------- (optional) QC dual-view panels for 3 random measured levels ----------
if QC_EXAMPLES > 0:
    # helper (re-uses the robust function to get QC payload)
    def qc_dual_full_and_roi_robust(ct_arr, ct_img, seg_arr, level_row, qc,
                                    window_full=(400,1800), window_roi=(500,2200)):
        sx, sy, sz = ct_img.GetSpacing()
        z=qc["z"]; y0,y1,x0,x1 = qc["y0"],qc["y1"],qc["x0"],qc["x1"]
        x_mid_r=qc["x_mid_r"]; y_ant_r=qc["y_ant_r"]; y_post_r=qc["y_post_r"]
        ct_roi=qc["ct_roi"]; seg_roi=qc["seg_roi"]; nonbone=qc["nonbone"]

        # full-FOV
        full=window_image(ct_arr[z], *window_full); seg_full=seg_arr[z]
        fig,axs=plt.subplots(1,2,figsize=(11,5.5))
        extent_full=[0, full.shape[1]*sx, full.shape[0]*sy, 0]
        axs[0].imshow(overlay_segmentation(full, seg_full), extent=extent_full); axs[0].axis('off'); axs[0].set_aspect('equal')
        axs[0].axvline((x0+x_mid_r)*sx, color='white', lw=1.2, ls='--'); axs[0].set_title(f"Full axial (z={z})")

        # ROI
        roi=window_image(ct_roi, *window_roi); extent_roi=[0, roi.shape[1]*sx, roi.shape[0]*sy, 0]
        axs[1].imshow(overlay_segmentation(roi, seg_roi), extent=extent_roi); axs[1].axis('off'); axs[1].set_aspect('equal')
        canal = np.ma.masked_where(nonbone==0, nonbone)
        axs[1].imshow(canal, extent=extent_roi, cmap='Greens', alpha=0.25, interpolation='nearest')
        x_mm=x_mid_r*sx
        axs[1].axvline(x_mm, color='white', lw=1.2, ls='--')
        axs[1].hlines([y_ant_r*sy, y_post_r*sy], x_mm-8, x_mm+8, colors=['cyan','yellow'], lw=2)
        # midline gap for display
        gap_px = 0
        H = nonbone.shape[0]
        for yy in range(int(y_post_r)+1, H):
            if nonbone[yy, int(x_mid_r)]: gap_px += 1
            else: break
        if gap_px>0:
            apd=gap_px*sy
            axs[1].hlines([(y_post_r+gap_px)*sy], x_mm-8, x_mm+8, colors='red', lw=2)
            axs[1].text(x_mm+6, (y_post_r*sy + (y_post_r+gap_px)*sy)/2,
                        f"APD(mid) ≈ {apd:.1f} mm", color='white',
                        bbox=dict(fc='black', alpha=0.55), fontsize=10)
        axs[1].set_title("Coned-in ROI (robust canal)")
        plt.suptitle(f"QC — {level_row['level']} (midline & ROI)", y=0.98, fontsize=12)
        plt.tight_layout(); plt.show()

    # sample levels with measurements
    candidates = cohort_eval.dropna(subset=["APD_final_mm"]).sample(min(QC_EXAMPLES, 
                        cohort_eval["APD_final_mm"].notna().sum()), random_state=RANDOM_SEED)
    print(f"[QC] Rendering {len(candidates)} dual-view panels...")
    for _, rr in candidates.iterrows():
        sid = rr["study_id"]; Ltxt = rr["level"]; upper=int(Ltxt.split('/')[0][1]); lower=int(Ltxt.split('/')[1][1])
        # reload study (simple and reliable for small QC count)
        p = [pp for pp in eligible if pp.name==sid][0]
        ct_img, ct_arr, _, _ = load_best_series_any(p)
        seg_img, seg_arr = load_seg_nifti(seg_map[p.name])
        same_geom = (ct_img.GetSize()==seg_img.GetSize() and
                     np.allclose(ct_img.GetSpacing(), seg_img.GetSpacing(), atol=1e-3) and
                     tuple(ct_img.GetDirection())==tuple(seg_img.GetDirection()))
        if not same_geom:
            seg_arr = sitk.GetArrayFromImage(resample_like(seg_img, ct_img, True))
        r0 = {"upper":upper, "lower":lower, "z_mid":int(rr["z_mid_used"]), "x_mid":int(rr["x_mid_used"])}
        m = measure_axial_robust(ct_arr, ct_img, seg_arr, **r0)
        if m and m["qc"]:
            r_level = {"level": Ltxt}
            qc_dual_full_and_roi_robust(ct_arr, ct_img, seg_arr, r_level, m["qc"])



# === Cell: MRI → CT transfer weights resolver (no training) ===
import os, glob, torch, timm

# ---- (A) optional: set your known MRI checkpoint path here (if you added a dataset with weights)
PREFER_USER_CKPT = "/kaggle/input/lumbar-model-weights/_scs_classify_5ch_axsagt2-lstm-mil_auxloss_auxdepth_convnext-s_for_exp0.ckpt"
# Leave as None to auto-search
if not os.path.exists(PREFER_USER_CKPT):
    PREFER_USER_CKPT = None

# ---- (B) auto-search common locations if not specified
SEARCH_GLOBS = [
    "/kaggle/input/*mri*weights*/*.pt",
    "/kaggle/input/*mri*weights*/*.pth",
    "/kaggle/input/*lumbar*weights*/*.pt",
    "/kaggle/input/*lumbar*weights*/*.pth",
    "/kaggle/input/*weights*/*.ckpt",
    "/kaggle/input/*mri*/*.ckpt",
]
def find_mri_ckpt():
    if PREFER_USER_CKPT and os.path.exists(PREFER_USER_CKPT):
        return PREFER_USER_CKPT
    for pat in SEARCH_GLOBS:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

# ---- (C) robust loader that adapts first conv to 1-channel and strips Lightning prefixes
def load_mri_transfer_into_backbone(backbone, ckpt_path, in_chans_expected=1):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[xfer] No MRI checkpoint found; using ImageNet weights already in timm.")
        return 0
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    # Keep only backbone.* if present; strip prefix
    cleaned = {}
    for k,v in sd.items():
        if "backbone" in k:
            k2 = k.split("backbone.",1)[-1]
            cleaned[k2] = v
        elif k in backbone.state_dict():
            cleaned[k] = v

    # First conv channel adaptation if needed (e.g., 5ch → 1ch)
    for k,v in list(cleaned.items()):
        if "weight" in k and v.ndim==4 and v.shape[1] != in_chans_expected:
            with torch.no_grad():
                if v.shape[1] > in_chans_expected:
                    v = v.mean(dim=1, keepdim=True).repeat(1, in_chans_expected, 1, 1)
                else:
                    v = v.repeat(1, in_chans_expected//v.shape[1] + 1, 1, 1)[:, :in_chans_expected]
            cleaned[k] = v

    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(missing)
    print(f"[xfer] Loaded {loaded} tensors from MRI ckpt: {os.path.basename(ckpt_path)} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return loaded

# ---- (D) build backbone and apply transfer (or ImageNet)
MODEL_NAME = "convnext_tiny.in12k_ft_in1k"   # good small backbone
IN_CHANS   = 1                               # we use single-channel axial slices
print(f"[model] Building {MODEL_NAME} (ImageNet weights) with in_chans={IN_CHANS}")
backbone = timm.create_model(MODEL_NAME, pretrained=True, in_chans=IN_CHANS, num_classes=0)

MRI_CKPT = find_mri_ckpt()
if MRI_CKPT:
    _ = load_mri_transfer_into_backbone(backbone, MRI_CKPT, in_chans_expected=IN_CHANS)
else:
    print("[xfer] MRI weights not found; proceeding with ImageNet initialization.")

# ---- (E) tiny sanity: forward a dummy slice to confirm the feature size
with torch.no_grad():
    dummy = torch.randn(1, IN_CHANS, 256, 256)
    feat = backbone(dummy)
    if feat.ndim == 4:
        feat = torch.nn.AdaptiveAvgPool2d(1)(feat).flatten(1)
    nf = feat.shape[1]
print(f"[model] Backbone feature dimension: {nf}")
# Save the backbone state_dict so downstream cells can reuse it quickly if needed
torch.save(backbone.state_dict(), "/kaggle/working/backbone_ct_init.pth")
print("[save] /kaggle/working/backbone_ct_init.pth")


















# After you have `meas_df` from 6H/6G
import numpy as np

meas_df["APD_final_mm"] = np.where(
    meas_df["APD_band_mm"].notna(), meas_df["APD_band_mm"], meas_df["APD_mid_mm"]
)

def sev_from_apd(a):
    if np.isnan(a): return -1
    return 2 if a < 10.0 else (1 if a < 13.0 else 0)  # severe <10, moderate 10–13
meas_df["severity_final"] = meas_df["APD_final_mm"].apply(sev_from_apd)

# OPTIONAL clamp: if APD_mid << APD_band by >6 mm (likely artifact), trust the band
meas_df["APD_final_mm"] = np.where(
    (meas_df["APD_mid_mm"].notna()) & (meas_df["APD_band_mm"].notna()) &
    ((meas_df["APD_band_mm"] - meas_df["APD_mid_mm"]) > 6.0),
    meas_df["APD_band_mm"], meas_df["APD_final_mm"]
)
meas_df["severity_final"] = meas_df["APD_final_mm"].apply(sev_from_apd)
display(meas_df[["level","APD_mid_mm","APD_band_mm","APD_final_mm","severity_mid","severity_band","severity_final"]])



# === Cell 7A: QC overlay for a chosen level (draw APD) ===
def qc_draw_level(ct_arr, ct_img, seg_arr, level_row, flip_ud=True):
    lvl = level_row["level"]; z_mid, x_mid = int(level_row["z_mid"]), int(level_row["x_mid"])
    sx, sy, sz = ct_img.GetSpacing()
    # build a sagittal band around z_mid for nicer context
    band = ct_arr[max(0,z_mid-2):min(ct_arr.shape[0], z_mid+3), :, x_mid].astype(np.float32)
    # collapse band to median for display
    sag_disp = np.median(band, axis=0)  # (Y,)
    # expand to 2D image for overlay (H, W) as (Zband, Y)
    sag2d = band  # (Hk, Y)
    # window & overlay vertebra labels projected to this slice column
    lbl_band = seg_arr[max(0,z_mid-2):min(ct_arr.shape[0], z_mid+3), :, x_mid]
    lbl_proj = (lbl_band.max(axis=0)).astype(np.int32)  # (Y,)

    # Construct 2D for overlay function: make HxW image (use H=band size)
    v = np.tile(sag_disp[None,:], (lbl_band.shape[0],1))
    ov = overlay_segmentation(window_image(v, 400, 1800), np.tile(lbl_proj[None,:], (lbl_band.shape[0],1)))

    # Compute edges for a single z within band (center slice)
    # inside qc_draw_level(...)
    res_vb = _posterior_body_from_mask(lbl_band, level_row["upper"])
    y_lam_ant = _lamina_anterior_y(band, res_vb[1]) if res_vb is not None else None

    fig, ax = plt.subplots(figsize=(6, 6 * (lbl_band.shape[0]*sz)/(lbl_proj.shape[0]*sy)))
    extent = [0, lbl_proj.shape[0]*sy, lbl_band.shape[0]*sz, 0]
    ax.imshow(ov, extent=extent)
    if res is not None:
        y_vb_ant, y_vb_post, y_lam_ant = res
        # draw vertical lines (y axis in image coordinate)
        yv = [y_vb_ant*sy, y_vb_post*sy, y_lam_ant*sy]
        for i,(c,lbl) in enumerate(zip(["cyan","yellow","red"], ["VB anterior","VB posterior","Lamina anterior"])):
            ax.axvline(yv[i], color=c, lw=2, label=lbl)
        apd_mm = max(0.0, yv[2]-yv[1])
        ax.text(5, 5, f"APD ≈ {apd_mm:.1f} mm", color="white", fontsize=11, ha="left", va="top",
                bbox=dict(facecolor="black", alpha=0.5))
        ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"QC — {lvl} (sagittal band at x={x_mid})")
    ax.set_xlabel("mm (anterior → posterior)"); ax.set_ylabel("mm (superior → inferior)")
    ax.axis('off'); plt.tight_layout(); plt.show()

# Draw QC for the first level that has a measurement
ok_rows = meas_df[meas_df["severity"] >= 0]
if len(ok_rows):
    qc_draw_level(ct_arr2, ct_img2, seg_arr2, ok_rows.iloc[0])
else:
    print("No measurable levels in this study.")



# === Cell 8A: Export + summary ===
OUT_DF = pd.DataFrame(meas_rows)
OUT_DF["study_id"] = study_dir2.name
cols = ["study_id","level","APD_mm","Torg","severity","z_mid","x_mid","upper","lower"]
OUT_DF = OUT_DF[cols]
csv_path = f"/kaggle/working/ccs_measurements_{study_dir2.name}.csv"
OUT_DF.to_csv(csv_path, index=False)
print(f"[SAVE] Wrote per-level measurements: {csv_path}")

# Summary bar (severity per level for this study)
sev_map = {0:"None/Mild",1:"Moderate",2:"Severe",-1:"N/A"}
OUT_DF["severity_txt"] = OUT_DF["severity"].map(sev_map)
ax = OUT_DF.plot(x="level", y="APD_mm", kind="bar", legend=False, figsize=(7,3.2))
ax.set_ylim(0, 25); ax.set_ylabel("AP canal diameter (mm)"); ax.set_xlabel("Level")
for p, s in zip(ax.patches, OUT_DF["severity_txt"]):
    ax.annotate(s, (p.get_x()+p.get_width()/2, p.get_height()+0.3), ha="center", fontsize=9)
plt.tight_layout(); plt.show()


