import os
import glob
import pickle
from collections import defaultdict, namedtuple

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from tqdm import tqdm

# =========================
# Configuration
# =========================
ROOT_DIR = "/kaggle/input/rsna-aneurysm/train"  # set to dataset root
OUT_DIR = "./preproc_rsna_aneurysm"             # outputs: pickles + npy volumes
os.makedirs(OUT_DIR, exist_ok=True)

# Options
LOAD_PIXEL_DATA = True           # set False if only indexing/metadata needed
APPLY_HU_CONVERSION = True       # convert to HU using RescaleSlope/Intercept
APPLY_WINDOWING = False          # typical CTA window, toggle as needed
WINDOW_CENTER = 40.0
WINDOW_WIDTH = 400.0
CLIP_TO_WINDOW = True            # clip to [WL-0.5, WH-0.5] after windowing
FLOAT32_OUTPUT = True            # store arrays as float32
SAVE_PER_SERIES_VOL = True       # save each 3D volume as .npy
SAVE_NEIGHBORS = True            # map neighboring slice SOPInstanceUIDs

# =========================
# Helpers
# =========================
SeriesKey = namedtuple("SeriesKey", ["study_uid", "series_uid"])


def window_image(img, center, width, clip=True):
    # Linear windowing
    low = center - width / 2.0
    high = center + width / 2.0
    img = (img - low) / (high - low)
    img = img * 255.0
    if clip:
        img = np.clip(img, 0, 255)
    return img


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def is_multiframe(ds: Dataset):
    # Enhanced CT can be multi-frame with PerFrameFunctionalGroupsSequence
    return hasattr(ds, "NumberOfFrames") and ds.NumberOfFrames is not None and ds.NumberOfFrames > 1


def extract_orientation_normal(iop):
    # iop length 6: row (0..2), col (3..5)
    r = np.array(iop[0:3], dtype=float)
    c = np.array(iop[3:6], dtype=float)
    n = np.cross(r, c)
    return n


def sort_by_geometry(file_list):
    # Robust geometric sorting using IOP+IPP; fallback to InstanceNumber; fallback to as-is
    headers = []
    for f in file_list:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        inst = getattr(ds, "InstanceNumber", None)
        headers.append((f, ipp, iop, inst))

    # Try geometric sort
    if all(h[1] is not None and h[2] is not None for h in headers):
        n = extract_orientation_normal(headers[2])
        positions = []
        for _, ipp, _, _ in headers:
            positions.append(np.dot(np.array(ipp, dtype=float), n))
        order = np.argsort(positions)
        return [headers[i] for i in order]

    # Fallback: InstanceNumber
    if all(h[3] is not None for h in headers):
        order = np.argsort([h[3] for h in headers])
        return [headers[i] for i in order]

    # Last resort: filesystem order
    return [h for h in headers]


def index_dicoms(root_dir):
    # Recursively find all .dcm and group by Study/Series
    series_map = defaultdict(list)
    for f in glob.glob(os.path.join(root_dir, "**", "*.dcm"), recursive=True):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if study_uid and series_uid:
                series_map[SeriesKey(study_uid, series_uid)].append(f)
        except Exception:
            # Ignore unreadable
            continue
    return series_map


def read_multiframe_volume(ds: Dataset):
    # Extract 3D array and per-frame geometry from multi-frame CT
    n_frames = int(ds.NumberOfFrames)
    rows = int(ds.Rows)
    cols = int(ds.Columns)

    # Pixel data read
    vol = ds.pixel_array  # shape: (frames, rows, cols) or (rows, cols, frames) depending on handler
    if vol.shape == rows and vol.shape[1] == cols and vol.shape[-1] == n_frames:
        # Make it (Z, Y, X)
        vol = np.moveaxis(vol, -1, 0)
    elif vol.shape == n_frames:
        # already (Z, Y, X)
        pass
    else:
        # Attempt to coerce
        vol = vol.reshape((n_frames, rows, cols))

    # Spacing
    pixsp = getattr(ds, "PixelSpacing", None)
    dz = safe_float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", np.nan)))
    dy = safe_float(pixsp) if pixsp is not None else np.nan
    dx = safe_float(pixsp[1]) if pixsp is not None else np.nan

    slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)

    meta = dict(
        modality=getattr(ds, "Modality", None),
        manufacturer=getattr(ds, "Manufacturer", None),
        convolution_kernel=str(getattr(ds, "ConvolutionKernel", "")),
        kVp=safe_float(getattr(ds, "KVP", np.nan)),
        exposure=safe_float(getattr(ds, "Exposure", np.nan)),
        pixel_spacing=(dy, dx, dz),
        rescale_slope=slope,
        rescale_intercept=intercept,
        series_description=str(getattr(ds, "SeriesDescription", "")),
        study_uid=getattr(ds, "StudyInstanceUID", None),
        series_uid=getattr(ds, "SeriesInstanceUID", None)
    )
    return vol, meta


def read_singleframe_volume(sorted_files):
    # Load slices into a 3D volume (Z, Y, X)
    slices = []
    metas = []
    for f in sorted_files:
        ds = pydicom.dcmread(f, force=True)
        arr = ds.pixel_array
        slices.append(arr)
        pixsp = getattr(ds, "PixelSpacing", None)
        dy = safe_float(pixsp) if pixsp is not None else np.nan
        dx = safe_float(pixsp[1]) if pixsp is not None else np.nan
        dz = safe_float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", np.nan)))
        slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        meta = dict(
            sop_uid=str(getattr(ds, "SOPInstanceUID", "")),
            instance_number=getattr(ds, "InstanceNumber", None),
            image_position=getattr(ds, "ImagePositionPatient", None),
            image_orientation=getattr(ds, "ImageOrientationPatient", None),
            pixel_spacing=(dy, dx),
            slice_thickness=safe_float(getattr(ds, "SliceThickness", np.nan)),
            spacing_between_slices=safe_float(getattr(ds, "SpacingBetweenSlices", np.nan)),
            rescale_slope=slope,
            rescale_intercept=intercept,
            convolution_kernel=str(getattr(ds, "ConvolutionKernel", "")),
            kVp=safe_float(getattr(ds, "KVP", np.nan)),
            exposure=safe_float(getattr(ds, "Exposure", np.nan)),
        )
        metas.append(meta)

    vol = np.stack(slices, axis=0)  # (Z, Y, X)

    # Attempt consistent spacing
    dy = metas["pixel_spacing"] if metas and metas["pixel_spacing"] == metas["pixel_spacing"] else np.nan
    dx = metas["pixel_spacing"][1] if metas and metas["pixel_spacing"][1] == metas["pixel_spacing"][1] else np.nan
    # Prefer SpacingBetweenSlices; fallback to SliceThickness; compute from IPP if needed
    dz = metas["spacing_between_slices"]
    if np.isnan(dz) or dz == 0:
        dz = metas["slice_thickness"]
    # IPP-based spacing if available
    z_positions = []
    has_ipp = all(m["image_position"] is not None for m in metas)
    has_iop = all(m["image_orientation"] is not None for m in metas)
    if has_ipp and has_iop:
        n = extract_orientation_normal(metas["image_orientation"])
        for m in metas:
            pos = np.dot(np.array(m["image_position"], dtype=float), n)
            z_positions.append(pos)
        z_positions = np.array(z_positions)
        diffs = np.diff(np.sort(z_positions))
        if diffs.size > 0:
            dz_est = float(np.median(np.abs(diffs)))
            if not np.isnan(dz_est) and dz_est > 0:
                dz = dz_est

    slope = metas["rescale_slope"] if metas else 1.0
    intercept = metas["rescale_intercept"] if metas else 0.0

    series_meta = dict(
        pixel_spacing=(dy, dx, dz),
        rescale_slope=slope,
        rescale_intercept=intercept,
        convolution_kernel=metas["convolution_kernel"] if metas else "",
        kVp=metas["kVp"] if metas else np.nan,
        exposure=metas["exposure"] if metas else np.nan,
        per_slice=metas
    )
    return vol, series_meta


def to_hounsfield(vol, slope, intercept):
    # Convert to HU in float32
    vol = vol.astype(np.float32)
    vol = vol * (slope if slope is not None and not np.isnan(slope) else 1.0)
    vol = vol + (intercept if intercept is not None and not np.isnan(intercept) else 0.0)
    return vol

def build_neighbors(sorted_files):
    # Return dict sop_uid -> (prev_uid, next_uid, filepath)
    uids = []
    for f in sorted_files:
        ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        uids.append((str(getattr(ds, "SOPInstanceUID", "")), f))
    n = len(uids)
    nb = {}
    for i, (uid, f) in enumerate(uids):
        prev_uid = uids[i-1] if i > 0 else uid
        next_uid = uids[i+1] if i < n-1 else uid
        nb[uid] = dict(prev_uid=prev_uid, next_uid=next_uid, path=f)
    return nb


# =========================
# Main pipeline
# =========================
def main():
    series_map = index_dicoms(ROOT_DIR)
    print(f"Found {len(series_map)} series")

    series_index = {}     # key: (study_uid, series_uid) -> metadata summary
    image_index = {}      # key: sop_uid -> metadata (neighbors, z proxy, exposure, thickness, path)
    saved_volumes = []    # paths to saved npy volumes

    for key, files in tqdm(series_map.items(), total=len(series_map)):
        # Detect multi-frame vs single-frame
        first_ds = pydicom.dcmread(files, stop_before_pixels=True, force=True)
        study_uid = getattr(first_ds, "StudyInstanceUID", None)
        series_uid = getattr(first_ds, "SeriesInstanceUID", None)
        series_desc = str(getattr(first_ds, "SeriesDescription", ""))

        if is_multiframe(first_ds):
            if not LOAD_PIXEL_DATA:
                # Index-only
                series_index[key] = dict(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_description=series_desc,
                    n_slices=int(first_ds.NumberOfFrames),
                    pixel_spacing=None,
                )
            else:
                ds_full = pydicom.dcmread(files, force=True)  # need PixelData
                vol, meta = read_multiframe_volume(ds_full)
                if APPLY_HU_CONVERSION:
                    vol = to_hounsfield(vol, meta["rescale_slope"], meta["rescale_intercept"])
                if APPLY_WINDOWING:
                    vol = window_image(vol, WINDOW_CENTER, WINDOW_WIDTH, clip=CLIP_TO_WINDOW)
                if FLOAT32_OUTPUT and vol.dtype != np.float32:
                    vol = vol.astype(np.float32)

                save_path = os.path.join(OUT_DIR, f"{study_uid}__{series_uid}.npy")
                if SAVE_PER_SERIES_VOL:
                    np.save(save_path, vol)
                    saved_volumes.append(save_path)

                # Build a minimal image_index using frame numbers
                n = vol.shape
                for i in range(n):
                    sop_uid = f"{series_uid}_frame_{i+1}"
                    prev_uid = f"{series_uid}_frame_{max(1, i)}"
                    next_uid = f"{series_uid}_frame_{min(n, i+2)}"
                    image_index[sop_uid] = dict(
                        study_id=study_uid,
                        series_id=series_uid,
                        image_minus1=prev_uid,
                        image_plus1=next_uid,
                        path=f"{files}#frame={i+1}",
                        z_pos=np.nan,
                        exposure=meta.get("exposure", np.nan),
                        thickness=meta.get("pixel_spacing", (np.nan, np.nan, np.nan))[2]
                    )

                series_index[key] = dict(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_description=series_desc,
                    n_slices=n,
                    pixel_spacing=meta["pixel_spacing"],
                    saved_path=save_path if SAVE_PER_SERIES_VOL else None
                )

        else:
            # Multi-file single-frame series
            sorted_files = sort_by_geometry(files)
            if SAVE_NEIGHBORS:
                nb = build_neighbors(sorted_files)
                image_index.update({
                    k: dict(
                        study_id=study_uid,
                        series_id=series_uid,
                        image_minus1=v["prev_uid"],
                        image_plus1=v["next_uid"],
                        path=v["path"],
                        z_pos=np.nan,  # filled later if IPP available
                        exposure=np.nan,
                        thickness=np.nan
                    ) for k, v in nb.items()
                })

            if not LOAD_PIXEL_DATA:
                series_index[key] = dict(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    series_description=series_desc,
                    n_slices=len(sorted_files),
                    pixel_spacing=None,
                )
                continue

            vol, meta = read_singleframe_volume(sorted_files)
            if APPLY_HU_CONVERSION:
                vol = to_hounsfield(vol, meta["rescale_slope"], meta["rescale_intercept"])
            if APPLY_WINDOWING:
                vol = window_image(vol, WINDOW_CENTER, WINDOW_WIDTH, clip=CLIP_TO_WINDOW)
            if FLOAT32_OUTPUT and vol.dtype != np.float32:
                vol = vol.astype(np.float32)

            # Update z_pos, exposure, thickness in image_index if neighbors computed
            if SAVE_NEIGHBORS and "per_slice" in meta:
                # derive z from IPP along normal if available
                zvals = []
                has_ipp = all(m.get("image_position") is not None for m in meta["per_slice"])
                has_iop = all(m.get("image_orientation") is not None for m in meta["per_slice"])
                nvec = None
                if has_ipp and has_iop:
                    nvec = extract_orientation_normal(meta["per_slice"][0]["image_orientation"])
                    for m in meta["per_slice"]:
                        zvals.append(float(np.dot(np.array(m["image_position"], dtype=float), nvec)))
                else:
                    zvals = [np.nan] * len(meta["per_slice"])

                for i, m in enumerate(meta["per_slice"]):
                    sop_uid = m["sop_uid"]
                    if sop_uid in image_index:
                        image_index[sop_uid]["z_pos"] = zvals[i]
                        image_index[sop_uid]["exposure"] = m["exposure"]
                        # Prefer spacing_between_slices; fallback to slice_thickness
                        thick = m["spacing_between_slices"]
                        if np.isnan(thick) or thick == 0:
                            thick = m["slice_thickness"]
                        image_index[sop_uid]["thickness"] = thick

            save_path = os.path.join(OUT_DIR, f"{study_uid}__{series_uid}.npy")
            if SAVE_PER_SERIES_VOL:
                np.save(save_path, vol)
                saved_volumes.append(save_path)

            dz = meta["pixel_spacing"][2] if meta["pixel_spacing"] else np.nan
            series_index[key] = dict(
                study_uid=study_uid,
                series_uid=series_uid,
                series_description=series_desc,
                n_slices=vol.shape,
                pixel_spacing=meta["pixel_spacing"],
                saved_path=save_path if SAVE_PER_SERIES_VOL else None
            )

    # Persist indices
    with open(os.path.join(OUT_DIR, "series_index.pickle"), "wb") as f:
        pickle.dump(series_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUT_DIR, "image_index.pickle"), "wb") as f:
        pickle.dump(image_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUT_DIR, "saved_volumes.pickle"), "wb") as f:
        pickle.dump(saved_volumes, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Series indexed: {len(series_index)}")
    print(f"Images indexed: {len(image_index)}")
    print(f"Volumes saved: {len(saved_volumes)}")

if __name__ == "__main__":
    main()




