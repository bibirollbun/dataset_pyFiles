# =========================
# Z-sweep on TRAIN (3 tomos) for Z ∈ {1,10,30}
# Prints per-tomogram results + total runtime per Z
# Saves: /kaggle/working/z_sweep_train_full.csv
#        /kaggle/working/z_sweep_train_table.csv
#        /kaggle/working/z_sweep_train_summary.csv
# =========================

# --- Basic setup ---
import os, sys, io, time, logging, warnings, re, random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

# Paths
DATA_ROOT  = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TRAIN_DIR  = f"{DATA_ROOT}/train"
GT_CSV     = f"{DATA_ROOT}/train_labels.csv"
MODEL_PATH = "/kaggle/input/mahf-yolo-train/mayolov2f.pt"

# Params
SIZE = 1024
CONFIDENCE_THRESHOLD = 0.8
WBF_IOU   = 0.5
NMS3D_IOU = 0.2
CONCENTRATION = 1.0      # 1.0 = use all slices
Z_LIST = [1]     # <<< sweep values

# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# --- Pull ultralytics fork used by MHAF-YOLO ---
!cp -r /kaggle/input/mhafyolo/pytorch/default/1/MHAF-YOLO-main /kaggle/working/
os.chdir("/kaggle/working/MHAF-YOLO-main")

from ultralytics import YOLOv10
from ultralytics.utils import LOGGER
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
LOGGER.setLevel(logging.ERROR)

class Quiet:
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = self._buf_out = io.StringIO()
        sys.stderr = self._buf_err = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._stdout, self._stderr

# --- Helpers ---
def list_tomos(root_dir: str):
    return [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

def clamp(v, a, b):
    return max(a, min(b, v))

def normalize_slice(gray_uint8: np.ndarray) -> np.ndarray:
    if gray_uint8.dtype != np.uint8:
        gray_uint8 = gray_uint8.astype(np.uint8)
    p2, p98 = np.percentile(gray_uint8, 2), np.percentile(gray_uint8, 98)
    if p98 <= p2 + 1e-6:
        return gray_uint8.copy()
    clipped = np.clip(gray_uint8, p2, p98)
    return (255.0 * (clipped - p2) / (p98 - p2)).astype(np.uint8)

def read_gray_resized(path: str, size: int) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        im = np.array(Image.open(path).convert("L"))
    if (im.shape[1], im.shape[0]) != (size, size):
        im = cv2.resize(im, (size, size))
    return im

def build_25d_rgb(tomo_dir: str, slice_files: list, idx: int, size: int, window: int) -> np.ndarray:
    n = len(slice_files)
    g = normalize_slice(read_gray_resized(os.path.join(tomo_dir, slice_files[idx]), size))
    prev = [read_gray_resized(os.path.join(tomo_dir, slice_files[clamp(idx-d,0,n-1)]), size)
            for d in range(1, window+1)]
    nxt  = [read_gray_resized(os.path.join(tomo_dir, slice_files[clamp(idx+d,0,n-1)]), size)
            for d in range(1, window+1)]
    r = normalize_slice(np.mean(prev, axis=0).astype(np.uint8)) if prev else g.copy()
    b = normalize_slice(np.mean(nxt,  axis=0).astype(np.uint8)) if nxt  else g.copy()
    return np.stack([r, g, b], axis=2)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

def weighted_box_fusion_simple(xyxy, conf, iou_thr=WBF_IOU):
    xyxy = np.asarray(xyxy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if xyxy.size == 0:
        return []
    if xyxy.ndim == 1:
        if xyxy.shape[0] != 4:
            return []
        xyxy = xyxy[None, :]
    if conf.ndim == 0:
        conf = np.array([float(conf)], dtype=np.float32)
    conf = conf.reshape(-1)
    if conf.shape[0] != xyxy.shape[0]:
        n = min(conf.shape[0], xyxy.shape[0])
        xyxy, conf = xyxy[:n], conf[:n]
        if n == 0:
            return []
    used = np.zeros(len(xyxy), dtype=bool)
    fused = []
    for i in range(len(xyxy)):
        if used[i]:
            continue
        group = [i]; used[i] = True
        for j in range(i+1, len(xyxy)):
            if used[j]:
                continue
            if iou_xyxy(xyxy[i], xyxy[j]) >= iou_thr:
                group.append(j); used[j] = True
        w = conf[group]; w = w / (w.sum() + 1e-9)
        bb = (xyxy[group] * w[:, None]).sum(axis=0)
        cf = float(conf[group].mean())
        fused.append((bb[0], bb[1], bb[2], bb[3], cf))
    return fused

def _safe_model_infer(model, image_rgb, img_size, device):
    if not isinstance(image_rgb, np.ndarray):
        image_rgb = np.array(image_rgb)
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)
    image_rgb = np.ascontiguousarray(image_rgb)
    with Quiet():
        return model(image_rgb, imgsz=img_size, device=device, verbose=False)

def predict_tta_wbf_single(model, image_rgb: np.ndarray, img_size=SIZE, conf_thres=0.0, wbf_iou=WBF_IOU, device='cuda:0'):
    H, W = image_rgb.shape[:2]
    all_boxes, all_confs, all_clss = [], [], []
    # original
    res = _safe_model_infer(model, image_rgb, img_size, device)
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        all_boxes.append(xyxy); all_confs.append(conf); all_clss.append(cls)
    # hflip
    img_f = cv2.flip(image_rgb, 1)
    resf = _safe_model_infer(model, img_f, img_size, device)
    for r in resf:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        inv = xyxy.copy()
        inv[:, 0] = W - xyxy[:, 2]
        inv[:, 2] = W - xyxy[:, 0]
        all_boxes.append(inv); all_confs.append(conf); all_clss.append(cls)
    if len(all_boxes) == 0:
        return []
    boxes_cat = np.concatenate(all_boxes, axis=0)
    confs_cat = np.concatenate(all_confs, axis=0)
    clss_cat  = np.concatenate(all_clss, axis=0)
    keep = confs_cat >= conf_thres
    boxes_cat, confs_cat, clss_cat = boxes_cat[keep], confs_cat[keep], clss_cat[keep]
    fused_all = []
    for c in np.unique(clss_cat):
        idx = np.where(clss_cat == c)[0]
        fused = weighted_box_fusion_simple(boxes_cat[idx], confs_cat[idx], iou_thr=wbf_iou)
        for (x1,y1,x2,y2,cf) in fused:
            fused_all.append((x1,y1,x2,y2,cf,int(c)))
    return fused_all

def perform_3d_nms(detections, iou_threshold=NMS3D_IOU):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    final = []
    box_size = 24
    dist_thr = box_size * iou_threshold
    def dist3(a,b):
        return np.sqrt((a['z']-b['z'])**2 + (a['y']-b['y'])**2 + (a['x']-b['x'])**2)
    while dets:
        best = dets.pop(0)
        final.append(best)
        dets = [d for d in dets if dist3(d, best) > dist_thr]
    return final

# return also meta for logging: status, num_slices, elapsed
def process_tomogram(root_dir, tomo_id, model, size=SIZE, device=device, window_z=10):
    t0 = time.time()
    tomo_dir = os.path.join(root_dir, tomo_id)
    slice_files_all = sorted([f for f in os.listdir(tomo_dir) if f.lower().endswith('.jpg')])
    if len(slice_files_all) == 0:
        elapsed = time.time() - t0
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1,
                '_status': 'NONE', '_slices': 0, '_elapsed': elapsed}

    h0, w0 = cv2.imread(os.path.join(tomo_dir, slice_files_all[0]), cv2.IMREAD_GRAYSCALE).shape[:2]
    scale_y, scale_x = h0 / float(size), w0 / float(size)

    if CONCENTRATION < 1.0:
        take = max(1, int(len(slice_files_all) * CONCENTRATION))
        sel = np.linspace(0, len(slice_files_all)-1, take)
        slice_idx = np.round(sel).astype(int).tolist()
    else:
        slice_idx = list(range(len(slice_files_all)))

    all_dets = []
    for i in slice_idx:
        rgb = build_25d_rgb(tomo_dir, slice_files_all, i, size=size, window=window_z)
        try: znum = int(Path(slice_files_all[i]).stem.split('_')[1])
        except Exception: znum = i
        fused = predict_tta_wbf_single(model, rgb, img_size=size, conf_thres=0.0, wbf_iou=WBF_IOU, device=device)
        for (x1,y1,x2,y2,cf,cl) in fused:
            if cf >= CONFIDENCE_THRESHOLD:
                xc = 0.5*(x1+x2); yc = 0.5*(y1+y2)
                all_dets.append({'z': int(round(znum)),
                                 'y': int(round(yc * scale_y)),
                                 'x': int(round(xc * scale_x)),
                                 'confidence': float(cf)})

    final_dets = perform_3d_nms(all_dets, NMS3D_IOU)
    status = "FOUND" if final_dets else "NONE"
    best = final_dets[0] if final_dets else {'z': -1, 'y': -1, 'x': -1}
    elapsed = time.time() - t0
    return {'tomo_id': tomo_id,
            'Motor axis 0': int(best['z']), 'Motor axis 1': int(best['y']), 'Motor axis 2': int(best['x']),
            '_status': status, '_slices': len(slice_idx), '_elapsed': elapsed}

def load_gt_points(gt_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    cols = [c.lower().strip().replace('_',' ') for c in df.columns]
    df.columns = cols
    if {'motor axis 0','motor axis 1','motor axis 2','tomo id'}.issubset(df.columns):
        out = df[['tomo id','motor axis 2','motor axis 1','motor axis 0']].copy()
        out.columns = ['tomo_id','gt_x','gt_y','gt_z']
        return out
    raise ValueError(f"Unrecognized GT header: {df.columns}")

# --- Load model once (quiet warmup) ---
with Quiet():
    model = YOLOv10(MODEL_PATH)
model.to(device)
try:
    model.model.float()
except Exception:
    pass
with Quiet():
    _ = model(np.zeros((SIZE,SIZE,3), np.uint8), imgsz=SIZE, device=device, verbose=False)


# # --- Build 3-sample set from TRAIN (intersection of disk & GT) ---
# gt_points = load_gt_points(GT_CSV)
# train_ids_disk = set(list_tomos(TRAIN_DIR))
# train_ids_gt   = set(gt_points["tomo_id"].unique())
# candidates     = sorted(list(train_ids_disk & train_ids_gt))
# if len(candidates) == 0:
#     raise RuntimeError("No train tomograms found with GT on disk.")

# if len(candidates) < 3:
#     print(f"WARNING: only {len(candidates)} train tomograms available with GT; using all of them.")
#     sampled_ids = candidates
# else:
#     sampled_ids = ['tomo_003acc', 'tomo_00e047', 'tomo_01a877']

# print("Selected train tomograms (count={}):".format(len(sampled_ids)))
# print(sampled_ids)

# # --- Run sweep for each Z and collect predictions (with per-tomo logging) ---
# all_preds = []
# overall_t0 = time.time()
# for z in Z_LIST:
#     rows = []
#     print(f"\n=== Running Z={z} on {len(sampled_ids)} tomograms ===")
#     z_t0 = time.time()

#     for tid in sampled_ids:
#         out = process_tomogram(TRAIN_DIR, tid, model, size=SIZE, device=device, window_z=z)
#         out['Z_setup'] = z
#         rows.append(out)

#         # per-tomogram print: pred vs GT + errors
#         gtr = gt_points[gt_points['tomo_id'] == tid]
#         if len(gtr):
#             gx, gy, gz = int(gtr['gt_x'].iloc[0]), int(gtr['gt_y'].iloc[0]), int(gtr['gt_z'].iloc[0])
#             px, py, pz = int(out['Motor axis 2']), int(out['Motor axis 1']), int(out['Motor axis 0'])
#             errx, erry, errz = abs(px-gx), abs(py-gy), abs(pz-gz)
#             l2 = ( (px-gx)**2 + (py-gy)**2 + (pz-gz)**2 )**0.5
#             print(f"[Z={z}] {tid} — {out['_status']} — {out['_slices']} slices — {out['_elapsed']:.1f}s | "
#                   f"pred=(x={px}, y={py}, z={pz})  gt=(x={gx}, y={gy}, z={gz})  "
#                   f"|err|=({errx},{erry},{errz})  L2={l2:.1f}")
#         else:
#             print(f"[Z={z}] {tid} — {out['_status']} — {out['_slices']} slices — {out['_elapsed']:.1f}s | GT MISSING")

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # total runtime for this Z
#     z_elapsed = time.time() - z_t0
#     print(f"=== Total runtime for Z={z}: {z_elapsed:.1f}s ===")

#     df = pd.DataFrame(rows)
#     all_preds.append(df)

# overall_elapsed = time.time() - overall_t0

# preds_all = pd.concat(all_preds, ignore_index=True).rename(columns={
#     "Motor axis 0":"pred_z",
#     "Motor axis 1":"pred_y",
#     "Motor axis 2":"pred_x"
# })

# # --- Merge with GT & compute errors ---
# merged = (preds_all.merge(gt_points, on='tomo_id', how='left')
#           .assign(err_x=lambda d: (d['pred_x']-d['gt_x']).abs(),
#                   err_y=lambda d: (d['pred_y']-d['gt_y']).abs(),
#                   err_z=lambda d: (d['pred_z']-d['gt_z']).abs()))
# merged['err_L2'] = np.sqrt((merged['pred_x']-merged['gt_x'])**2 +
#                            (merged['pred_y']-merged['gt_y'])**2 +
#                            (merged['pred_z']-merged['gt_z'])**2)

# # --- Human-readable table: GT & preds per Z per tomo ---
# Z_LIST_SORTED = sorted(merged['Z_setup'].unique().tolist())
# rows_tbl = []
# for tid, g in merged.groupby('tomo_id'):
#     row = {'tomo_id': tid,
#            'GT(x,y,z)': (int(g['gt_x'].iloc[0]), int(g['gt_y'].iloc[0]), int(g['gt_z'].iloc[0]))}
#     for z in Z_LIST_SORTED:
#         gi = g[g['Z_setup']==z].iloc[0]
#         row[f'Pred@Z={z} (x,y,z)'] = (int(gi['pred_x']), int(gi['pred_y']), int(gi['pred_z']))
#         row[f'|err|@Z={z} (x,y,z)'] = (int(abs(gi['pred_x']-gi['gt_x'])),
#                                        int(abs(gi['pred_y']-gi['gt_y'])),
#                                        int(abs(gi['pred_z']-gi['gt_z'])))
#     rows_tbl.append(row)
# final_table = pd.DataFrame(rows_tbl)

# # --- Summary per Z ---
# summary = (merged.groupby('Z_setup')
#            .agg(mean_err_x=('err_x','mean'),
#                 mean_err_y=('err_y','mean'),
#                 mean_err_z=('err_z','mean'),
#                 mean_L2=('err_L2','mean'))
#            .round(3)
#            .reset_index())

# # --- Save CSVs ---
# out_dir = "/kaggle/working"
# full_csv    = f"{out_dir}/z_sweep_train_full.csv"
# table_csv   = f"{out_dir}/z_sweep_train_table.csv"
# summary_csv = f"{out_dir}/z_sweep_train_summary.csv"

# merged.to_csv(full_csv, index=False)
# final_table.to_csv(table_csv, index=False)
# summary.to_csv(summary_csv, index=False)

# print("\n=== Z-sweep summary (mean absolute error) ===")
# print(summary.to_string(index=False))
# print("\nSaved:")
# print(full_csv)
# print(table_csv)
# print(summary_csv)

# print(f"\n=== Overall runtime (all Z values): {overall_elapsed:.1f}s ===")


# 2) --- Submission generation section ---
TEST_DIR = f"{DATA_ROOT}/test"
SUBMISSION_PATH = "/kaggle/working/submission.csv"


def generate_submission(test_dir=TEST_DIR, model=model, submission_path=SUBMISSION_PATH, z_for_submission=None):
    """
    Runs inference on the test set and saves submission.csv
    """
    if z_for_submission is None:
        z_for_submission = Z_LIST[0]  # default to first sweep Z value
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    print(f"\n=== Generating submission on {total_tomos} test tomograms (Z={z_for_submission}) ===")

    results = []
    motors_found = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sequential loop 
    for tomo_id in test_tomos:
        result = process_tomogram(test_dir, tomo_id, model, size=SIZE, device=device, window_z=z_for_submission)
        result['Z_setup'] = z_for_submission
        results.append(result)

        has_motor = not pd.isna(result['Motor axis 0']) and result['Motor axis 0'] != -1
        if has_motor:
            motors_found += 1
        #     print(f"Motor found in {tomo_id} at position: "
        #           f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
        # else:
        #     print(f"No motor detected in {tomo_id}")

        # print(f"Current detection rate: {motors_found}/{len(results)} "
        #       f"({motors_found/len(results)*100:.1f}%)")

    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")
    print("\nSubmission preview:")
    print(submission_df.head())

    return submission_df


generate_submission(z_for_submission=Z_LIST[0])

