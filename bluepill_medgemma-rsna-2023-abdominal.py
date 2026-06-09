!pip install -U transformers


# %% [code] Config

from __future__ import annotations

import os
import re
import json
import warnings
from typing import Any
import torch

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm.auto import tqdm

from transformers import pipeline
from transformers.utils import logging as hf_logging
from huggingface_hub import login
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

CONFIG = {
    # HF / auth
    "USE_KAGGLE_SECRET": True,
    "KAGGLE_HF_SECRET_NAME": "HF_read",
    "HF_TOKEN": "YOUR_HF_TOKEN_HERE",
    "MODEL_NAME": "google/medgemma-1.5-4b-it",

    # data
    "SELECTION_CSV": "/kaggle/input/datasets/bluepill/abdominal-submission-250t-77a/submission_250t_77a.csv",
    "TRAIN_LABELS_CSV": "/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection/train_2024.csv",
    "DICOM_ROOT": "/kaggle/input/competitions/rsna-2023-abdominal-trauma-detection/train_images",
    "WORK_DIR": "/kaggle/working/medgemma_abdominal_dicoms",

    # model prompt
    "PROMPT": """\
Task: classify abdominal CT slice for traumatic injuries of the liver, spleen, kidneys, and bowel (blunt trauma).

Visibility rule:
- Evaluate kidneys, liver, and spleen ONLY if the organ (or a clear portion of it) is actually visible on THIS slice.
  If an organ is not visible on the slice, do NOT search for or report an abnormality in that organ.

A) Count as ANOMALY only if this slice shows one or more of the following traumatic findings clearly visible on this slice:
   1) Active contrast extravasation (jet/collection not conforming to a vessel or collecting system).
   2) Parenchymal laceration or deep contusion of liver/spleen/kidney.
   3) Subcapsular or periorgan hematoma.
   4) Devascularization/infarct zone in a target organ.
   5) Bowel wall discontinuity, focal full-thickness defect, or unequivocal traumatic wall thickening/hematoma.
   6) Free intraperitoneal air (pneumoperitoneum) or hemoperitoneum/free intraperitoneal fluid attributable to trauma.
   7) Mesenteric hematoma.

Steps:
1) Inspect liver, spleen, both kidneys, and bowel
2) If no abnormalities are visible, output label: normal.
3) If any abnormality is suspected, output label: anomaly.

Output format (MUST be exact, lowercase, no extra text):
label: normal
OR
label: anomaly
""".strip(),

    # axes
    "AXES_TO_EVAL": ["axial"],

    # intensity window
    "HU_CENTER": 40,
    "HU_WIDTH": 400,

    # slice selection
    "USE_CENTER_CROP_FOR_LONG_STUDIES": False,
    "CENTER_CROP_THRESHOLD": 300,
    "CENTER_FRACTION_LOW": 0.2,
    "CENTER_FRACTION_HIGH": 0.8,

    # study-level decisions
    "FRACTION_THRESHOLD": 0.10,
    "CONSECUTIVE_MIN_RUN": 3,
    "N_PARTS": 10,
    "PART_THRESHOLD": 0.25,
    "N_SAMPLES": 77
}

LABEL_RE = re.compile(r"label\s*:\s*(normal|anomaly)", re.IGNORECASE)
AXES_ALL = ("axial", "sagittal", "coronal")

os.makedirs(CONFIG["WORK_DIR"], exist_ok=True)



# Functions

def get_hf_token_from_config() -> str:
    if CONFIG.get("USE_KAGGLE_SECRET", False):
        if UserSecretsClient is None:
            raise RuntimeError("Kaggle secrets are unavailable. Set USE_KAGGLE_SECRET=False and fill HF_TOKEN.")
        return UserSecretsClient().get_secret(CONFIG["KAGGLE_HF_SECRET_NAME"])
    return CONFIG["HF_TOKEN"]


def build_medgemma_pipeline():
    print("Getting HF token...")
    hf_token = get_hf_token_from_config()
    login(token=hf_token)

    print(f"Loading model: {CONFIG['MODEL_NAME']}")
    pipe = pipeline(
        "image-text-to-text",
        model=CONFIG["MODEL_NAME"],
        trust_remote_code=True,
        device_map="auto",
    )
    print("Model loaded.")
    return pipe



def list_dicom_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and not f.startswith(".")
    ]


def choose_series_with_max_dicoms(patient_root: str) -> tuple[str | None, int]:
    if not os.path.isdir(patient_root):
        return None, 0

    subdirs = [
        os.path.join(patient_root, d)
        for d in os.listdir(patient_root)
        if os.path.isdir(os.path.join(patient_root, d))
    ]

    if not subdirs:
        n_files = len(list_dicom_files(patient_root))
        return (patient_root, n_files) if n_files > 0 else (None, 0)

    counts = [(subdir, len(list_dicom_files(subdir))) for subdir in subdirs]
    counts = [x for x in counts if x[1] > 0]
    if not counts:
        return None, 0

    best_dir, best_n = max(counts, key=lambda x: (x[1], x[0]))
    return best_dir, best_n


def _to_hu(ds) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        hu = -hu
    return hu


def load_dicom_series(series_dir: str) -> np.ndarray:
    files = list_dicom_files(series_dir)
    if not files:
        raise RuntimeError(f"No DICOM files in {series_dir}")

    dsets = []
    for p in files:
        try:
            ds = pydicom.dcmread(p, force=True, stop_before_pixels=False)
            _ = ds.pixel_array
            dsets.append(ds)
        except Exception:
            continue

    if not dsets:
        raise RuntimeError(f"No readable DICOMs in {series_dir}")

    ds0 = dsets[0]
    iop = getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    row_dir = np.array(iop[:3], dtype=np.float64)
    col_dir = np.array(iop[3:], dtype=np.float64)
    normal = np.cross(row_dir, col_dir)

    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            ipp = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
            return float(ipp @ normal)
        return int(getattr(ds, "InstanceNumber", 0))

    dsets = sorted(dsets, key=sort_key)

    h = int(getattr(ds0, "Rows"))
    w = int(getattr(ds0, "Columns"))
    z = len(dsets)

    volume = np.zeros((h, w, z), dtype=np.float32)
    for i, ds in enumerate(dsets):
        volume[:, :, i] = _to_hu(ds)

    return volume


def select_step_abdominal(n_slices: int) -> int:
    if n_slices < 50:
        return 1
    if n_slices < 100:
        return 2
    if n_slices < 200:
        return 4
    if n_slices < 400:
        return 6
    if n_slices < 600:
        return 8
    return 10


def choose_indices_for_stack(n_total: int) -> tuple[list[int], dict[str, Any]]:
    if n_total <= 0:
        raise ValueError("Empty study")

    use_center_crop = (
        bool(CONFIG["USE_CENTER_CROP_FOR_LONG_STUDIES"])
        and n_total > int(CONFIG["CENTER_CROP_THRESHOLD"])
    )

    if use_center_crop:
        start = int(n_total * float(CONFIG["CENTER_FRACTION_LOW"]))
        end = n_total - int(n_total * (1.0 - float(CONFIG["CENTER_FRACTION_HIGH"])))
        end = max(end, start + 1)
        n_eff = end - start
        step = select_step_abdominal(n_eff)
        indices = list(range(start, end, step))
    else:
        start = 0
        end = n_total
        step = select_step_abdominal(n_total)
        indices = list(range(0, n_total, step))

    meta = {
        "n_slices": n_total,
        "n_used": len(indices),
        "start": start,
        "end": end,
        "step": step,
    }
    return indices, meta


def extract_slice(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":
        return volume[:, :, idx]
    if axis == "sagittal":
        return volume[:, idx, :]
    if axis == "coronal":
        return volume[idx, :, :]
    raise ValueError(f"Unsupported axis: {axis}")


def hu_to_u8(slice_2d: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    s = np.clip(slice_2d, lo, hi)
    img = (s - lo) / (hi - lo + 1e-6)
    return (img * 255.0).astype(np.uint8)


def prepare_slices_for_axis(
    volume: np.ndarray,
    axis: str,
    center: float,
    width: float,
) -> tuple[list[int], list[Image.Image], dict[str, Any]]:
    h, w, d = volume.shape

    if axis == "axial":
        n_slices = d
        rotate180 = False
    elif axis == "sagittal":
        n_slices = w
        rotate180 = True
    elif axis == "coronal":
        n_slices = h
        rotate180 = True
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    indices, meta = choose_indices_for_stack(n_slices)

    images = []
    for i in indices:
        sl = extract_slice(volume, axis, i)
        if rotate180:
            sl = np.rot90(sl, 2)
        u8 = hu_to_u8(sl, center=center, width=width)
        images.append(Image.fromarray(u8, mode="L").convert("RGB"))

    return indices, images, meta


def prepare_slices_all_axes(
    volume: np.ndarray,
    axes: list[str],
    center: float,
    width: float,
) -> dict[str, dict[str, Any]]:
    out = {}
    for axis in axes:
        if axis not in AXES_ALL:
            raise ValueError(f"axis must be in {AXES_ALL}, got {axis!r}")
        indices, images, meta = prepare_slices_for_axis(volume, axis, center, width)
        out[axis] = {
            "indices": indices,
            "images": images,
            "meta": meta,
        }
    return out


def parse_label(text: str) -> int | None:
    if not isinstance(text, str):
        return None
    match = LABEL_RE.search(text)
    if not match:
        return None
    return 1 if match.group(1).lower() == "anomaly" else 0


def classify_slices_with_outputs(
    images,
    pipe_obj,
    prompt,
    axis,
    slice_indices,
    patient_id,
    series_id,
):
    rows = []

    for idx, image in tqdm(
        list(zip(slice_indices, images)),
        total=len(images),
        desc=f"{patient_id}/{series_id}/{axis}",
        leave=False,
    ):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        try:
            output = pipe_obj(text=messages, max_new_tokens=256)
            raw_text = output[0]["generated_text"][-1]["content"]
            parsed_label = parse_label(raw_text)
            status = "ok" if parsed_label is not None else "unparsed"
        except Exception as exc:
            print(f"[ERROR] patient={patient_id} series={series_id} axis={axis} slice={idx}")
            raise exc

        rows.append(
            {
                "patient_id": patient_id,
                "series_id": series_id,
                "axis": axis,
                "slice_index": int(idx),
                "raw_response": raw_text,
                "parsed_label": parsed_label,
                "status": status,
            }
        )

    return rows



def decision_by_fraction(labels: list[int], threshold: float) -> bool:
    if not labels:
        return False
    return sum(labels) / len(labels) > threshold


def decision_by_consecutive(labels: list[int], min_run: int) -> bool:
    if min_run <= 0 or not labels:
        return False
    run = 0
    for x in labels:
        if x == 1:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 0
    return False


def decision_by_parts(labels: list[int], n_parts: int, part_threshold: float) -> bool:
    if not labels or n_parts <= 0:
        return False
    size = len(labels)
    part_size = max(1, size // n_parts)

    for p in range(n_parts):
        start = p * part_size
        end = min(start + part_size, size) if p < n_parts - 1 else size
        if start >= end:
            continue
        part = labels[start:end]
        if sum(part) / len(part) > part_threshold:
            return True

    return False


def apply_study_decisions(labels: list[int]) -> dict[str, bool]:
    return {
        "decision_by_fraction": decision_by_fraction(labels, CONFIG["FRACTION_THRESHOLD"]),
        "decision_by_consecutive": decision_by_consecutive(labels, CONFIG["CONSECUTIVE_MIN_RUN"]),
        "decision_by_parts": decision_by_parts(labels, CONFIG["N_PARTS"], CONFIG["PART_THRESHOLD"]),
    }


def score_one_study(patient_id: str, series_dir: str, pipe_obj):
    series_id = os.path.basename(series_dir.rstrip("/"))
    volume = load_dicom_series(series_dir)

    slices_by_axis = prepare_slices_all_axes(
        volume=volume,
        axes=CONFIG["AXES_TO_EVAL"],
        center=CONFIG["HU_CENTER"],
        width=CONFIG["HU_WIDTH"],
    )

    axis_labels = {}
    axis_indices = {}
    all_labels = []
    slice_rows = []

    for axis in CONFIG["AXES_TO_EVAL"]:
        axis_data = slices_by_axis.get(axis, {})
        indices = list(axis_data.get("indices", []))
        images = list(axis_data.get("images", []))
        axis_indices[axis] = indices

        if not images:
            axis_labels[axis] = []
            continue

        rows = classify_slices_with_outputs(
            images=images,
            pipe_obj=pipe_obj,
            prompt=CONFIG["PROMPT"],
            axis=axis,
            slice_indices=indices,
            patient_id=patient_id,
            series_id=series_id,
        )
        slice_rows.extend(rows)

        labels = [int(x["parsed_label"]) for x in rows if x["parsed_label"] is not None]
        axis_labels[axis] = labels
        all_labels.extend(labels)

    decisions = apply_study_decisions(all_labels)
    anomaly_share = float(sum(all_labels) / len(all_labels)) if all_labels else 0.0

    h, w, d = volume.shape
    available_slices = {
        "axial": int(d),
        "sagittal": int(w),
        "coronal": int(h),
    }
    used_slices = {axis: len(axis_indices.get(axis, [])) for axis in CONFIG["AXES_TO_EVAL"]}

    study_row = {
        "patient_id": patient_id,
        "series_id": series_id,
        "series_dir": series_dir,
        "axes": CONFIG["AXES_TO_EVAL"],
        "volume_shape_hwd": [int(h), int(w), int(d)],
        "available_slices": available_slices,
        "used_slices": used_slices,
        "axis_indices": axis_indices,
        "axis_labels": axis_labels,
        "decisions": decisions,
        "anomaly_share": anomaly_share,
    }
    return study_row, slice_rows



# %% [code] Run

os.makedirs(CONFIG["WORK_DIR"], exist_ok=True)

if "train_df" in globals():
    labels_df = train_df.copy()
else:
    labels_df = pd.read_csv(CONFIG["TRAIN_LABELS_CSV"])

labels_df["patient_id"] = labels_df["patient_id"].astype(str)

select_df = pd.read_csv(CONFIG["SELECTION_CSV"])
select_df["patient_id"] = select_df["patient_id"].astype(str)

if "any_injury" not in select_df.columns:
    select_df = select_df.merge(
        labels_df[["patient_id", "any_injury"]],
        on="patient_id",
        how="left",
    )

select_df["any_injury"] = select_df["any_injury"].astype(int)
select_df = select_df.sort_values("patient_id").reset_index(drop=True)

neg_df = select_df[select_df["any_injury"] == 0].head(CONFIG['N_SAMPLES']).copy()
pos_df = select_df[select_df["any_injury"] == 1].head(CONFIG['N_SAMPLES']).copy()
eval_patients_df = pd.concat([neg_df, pos_df], ignore_index=True)

plan_rows = []
missing_patients = []

for _, row in eval_patients_df.iterrows():
    patient_id = str(row["patient_id"])
    patient_root = os.path.join(CONFIG["DICOM_ROOT"], patient_id)
    series_dir, n_dicoms = choose_series_with_max_dicoms(patient_root)

    if series_dir is None:
        missing_patients.append(patient_id)
        continue

    plan_rows.append(
        {
            "patient_id": patient_id,
            "target": int(row["any_injury"]),
            "series_id": os.path.basename(series_dir.rstrip("/")),
            "series_dir": series_dir,
            "n_dicoms": int(n_dicoms),
        }
    )

plan_df = pd.DataFrame(plan_rows)

print("Requested patients:", len(eval_patients_df))
print("Resolved studies:", len(plan_df))
if missing_patients:
    print("Missing patients:", len(missing_patients))

pipe_obj = build_medgemma_pipeline()

study_rows = []
slice_rows_all = []

for _, row in tqdm(plan_df.iterrows(), total=len(plan_df), desc="Abdominal DICOM studies"):
    study_row, slice_rows = score_one_study(
        patient_id=str(row["patient_id"]),
        series_dir=str(row["series_dir"]),
        pipe_obj=pipe_obj,
    )
    study_row["target"] = int(row["target"])
    study_rows.append(study_row)
    slice_rows_all.extend(slice_rows)

study_results_df = pd.DataFrame(study_rows)
slice_results_df = pd.DataFrame(slice_rows_all)

plan_csv = os.path.join(CONFIG["WORK_DIR"], "plan_154.csv")
study_csv = os.path.join(CONFIG["WORK_DIR"], "study_outputs_154.csv")
slice_csv = os.path.join(CONFIG["WORK_DIR"], "slice_outputs_154.csv")

plan_df.to_csv(plan_csv, index=False)
study_results_df.to_csv(study_csv, index=False)
slice_results_df.to_csv(slice_csv, index=False)

print("Saved:")
print(plan_csv)
print(study_csv)
print(slice_csv)

display(plan_df.head())
display(study_results_df.head())
display(slice_results_df.head())



# %% [code] Metrics

merged_df = labels_df[["patient_id", "any_injury"]].merge(
    study_results_df[["patient_id", "series_id", "anomaly_share", "target"]],
    on="patient_id",
    how="inner",
)

y_true = merged_df["any_injury"].astype(float)
y_score = merged_df["anomaly_share"].astype(float)

roc = roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else np.nan

best_f1 = 0.0
best_thr = 0.5
best_prec = 0.0
best_rec = 0.0

for thr in np.linspace(0, 1, 101):
    y_pred = (y_score >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    if f1 > best_f1:
        best_f1 = float(f1)
        best_thr = float(thr)
        best_prec = float(prec)
        best_rec = float(rec)

metrics_df = pd.DataFrame(
    [
        {
            "metric": "any_injury_overall",
            "roc_auc": roc,
            "best_threshold_f1": best_thr,
            "precision_at_best_f1": best_prec,
            "recall_at_best_f1": best_rec,
            "f1_at_best_f1": best_f1,
            "n": len(merged_df),
            "positives": int(y_true.sum()),
        }
    ]
)

metrics_csv = os.path.join(CONFIG["WORK_DIR"], "metrics_154.csv")
merged_csv = os.path.join(CONFIG["WORK_DIR"], "merged_eval_154.csv")

metrics_df.to_csv(metrics_csv, index=False)
merged_df.to_csv(merged_csv, index=False)

display(metrics_df)

print("Saved:")
print(metrics_csv)
print(merged_csv)


