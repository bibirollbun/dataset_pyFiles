# ============================================================
# Kaggle: Server-centric FL+KD (WBF) for YOLO Student (FIXED)
# - Prepares dataset from COCO + optional ImageNet LOC
# - Server-centric: client-forward -> WBF -> mix with Teacher -> pseudo-label -> train Student (1 epoch/round)
# - Visualization of Teacher vs Student vs WBF
# - Fixes:
#   * Removed invalid "with cv2.imread(...)" (use PIL.Image.open)
#   * Replaced truthiness checks on arrays/lists with len(...) > 0
# ============================================================


!pip -q install "ultralytics>=8.3.0" ensemble-boxes pycocotools opencv-python-headless



import os, sys, random, time, shutil, glob, math, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import yaml
import matplotlib.pyplot as plt
from PIL import Image  # <-- needed for ImageNet W,H

import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

try:
    from pycocotools.coco import COCO
except Exception as e:
    print("pycocotools import failed:", e)
    raise



SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DATASET_DIR = '/kaggle/working/dataset'
IMAGES_DIR  = os.path.join(DATASET_DIR, 'images')
LABELS_DIR  = os.path.join(DATASET_DIR, 'labels')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

NAMES = ['person', 'phone', 'reflex_camera', 'polaroid_camera']
NC    = len(NAMES)

TEACHER_PATH = '/kaggle/input/finetuneyoloteacher/pytorch/default/6/teacher_best.pt'  # <-- update if needed
STUDENT_VARIANT = 'n'  # 'n'|'s'|'m'

# FL+KD schedule
NUM_ROUNDS = 3
NUM_CLIENTS = 5
IMAGES_PER_ROUND = 1200
IMG_SIZE = 640
ALPHA_START = 1.0
ALPHA_END   = 0.5

FLKD_OUTDIR = '/kaggle/working/flkd'
os.makedirs(FLKD_OUTDIR, exist_ok=True)



def detect_coco_root():
    candidates = [
        '/kaggle/input/coco-2017-dataset/coco2017',
        '/kaggle/input/coco-2017-dataset',
        '/kaggle/input/coco2017',
        '/kaggle/input/coco-2017'
    ]
    for root in candidates:
        ann = os.path.join(root, 'annotations', 'instances_train2017.json')
        imgd = os.path.join(root, 'train2017')
        if os.path.exists(ann) and os.path.isdir(imgd):
            return root
    for dirpath, dirnames, filenames in os.walk('/kaggle/input'):
        if 'instances_train2017.json' in filenames:
            root = os.path.dirname(dirpath) if os.path.basename(dirpath) == 'annotations' else dirpath
            if os.path.isdir(os.path.join(root, 'train2017')):
                return root
    return None

def detect_imagenet_root():
    root = '/kaggle/input/imagenet-object-localization-challenge'
    return root if os.path.isdir(os.path.join(root, 'ILSVRC')) else None

COCO_ROOT = detect_coco_root()
IMAGENET_ROOT = detect_imagenet_root()
print(f"COCO root: {COCO_ROOT}")
print(f"ImageNet root: {IMAGENET_ROOT}")



def prepare_dataset():
    """
    Build YOLO-style dataset under /kaggle/working/dataset
    images/ and labels/ with 4 classes: person, phone, reflex_camera, polaroid_camera
    """
    class_counts = {0:0, 1:0, 2:0, 3:0}

    # ---- ImageNet LOC -> classes 2 & 3 (optional)
    if IMAGENET_ROOT:
        print("\n=== Processing ImageNet LOC ===")
        try:
            annotations_file = os.path.join(IMAGENET_ROOT, 'LOC_train_solution.csv')
            if os.path.exists(annotations_file):
                annotations = pd.read_csv(annotations_file)
                imagenet_classes = {
                    'n02992529': 1, # mobile phone -> phone
                    'n04069434': 2, # reflex camera
                    'n03976467': 3  # polaroid camera
                }
                desired = list(imagenet_classes.keys())

                filtered = annotations[annotations['PredictionString'].str.contains('|'.join(desired))]
                print(f"Found {len(filtered)} candidate records from ImageNet")

                max_per_class = {1: 1500, 2: 1000, 3: 1000}

                for _, row in tqdm(filtered.iterrows(), total=len(filtered)):
                    image_id = row['ImageId']
                    predictions = row['PredictionString'].split()
                    i = 0
                    wrote_any = False

                    label_path = os.path.join(LABELS_DIR, f'imagenet_{image_id}.txt')
                    lines = []
                    while i < len(predictions):
                        if predictions[i] in desired:
                            class_id = predictions[i]
                            xmin, ymin, xmax, ymax = map(float, predictions[i+1:i+5])

                            img_path = os.path.join(IMAGENET_ROOT, 'ILSVRC', 'Data', 'CLS-LOC', 'train', class_id, f'{image_id}.JPEG')
                            if not os.path.exists(img_path):
                                i += 5
                                continue

                            yolo_cls = imagenet_classes[class_id]
                            if class_counts[yolo_cls] >= max_per_class.get(yolo_cls, 10**9):
                                i += 5
                                continue

                            # Copy image once
                            out_img = os.path.join(IMAGES_DIR, f'imagenet_{image_id}.jpg')
                            if not os.path.exists(out_img):
                                shutil.copy(img_path, out_img)

                            # Get width/height using PIL (NO cv2 context manager!)
                            with Image.open(img_path) as im:
                                W, H = im.size

                            x_center = ((xmin + xmax) / 2) / W
                            y_center = ((ymin + ymax) / 2) / H
                            w_norm   = (xmax - xmin) / W
                            h_norm   = (ymax - ymin) / H

                            lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                            class_counts[yolo_cls] += 1
                            wrote_any = True
                            i += 5
                        else:
                            i += 1

                    if wrote_any:
                        mode = 'w' if not os.path.exists(label_path) else 'a'
                        with open(label_path, mode) as f:
                            f.writelines(lines)
            else:
                print("ImageNet LOC csv not found -> skip.")
        except Exception as e:
            print("ImageNet processing failed:", e)

    # ---- COCO -> classes 0 & 1
    if COCO_ROOT:
        print("\n=== Processing COCO ===")
        try:
            coco = COCO(os.path.join(COCO_ROOT, 'annotations', 'instances_train2017.json'))
            person_ids = coco.getImgIds(catIds=[1])   # person
            phone_ids  = coco.getImgIds(catIds=[77])  # cell phone
            max_person_images = 2000
            if len(person_ids) > max_person_images:
                person_ids = random.sample(person_ids, max_person_images)
            all_ids = list(set(person_ids + phone_ids))
            print(f"Person imgs: {len(person_ids)} | Phone imgs: {len(phone_ids)} | Unique: {len(all_ids)}")

            for img_id in tqdm(all_ids):
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[1,77])
                anns = coco.loadAnns(ann_ids)
                if not anns:
                    continue

                src_img = os.path.join(COCO_ROOT, 'train2017', img_info['file_name'])
                if not os.path.exists(src_img):
                    continue

                dst_img = os.path.join(IMAGES_DIR, f"coco_{img_info['file_name']}")
                if not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)

                lbl_path = os.path.join(LABELS_DIR, f"coco_{Path(img_info['file_name']).stem}.txt")
                with open(lbl_path, 'w') as f:
                    for ann in anns:
                        if ann['category_id'] == 1:
                            ycls = 0
                        elif ann['category_id'] == 77:
                            ycls = 1
                        else:
                            continue
                        x,y,w,h = ann['bbox']
                        if w <= 0 or h <= 0:
                            continue
                        x_c = (x + w/2) / img_info['width']
                        y_c = (y + h/2) / img_info['height']
                        w_n = w / img_info['width']
                        h_n = h / img_info['height']
                        x_c = max(0,min(1,x_c)); y_c = max(0,min(1,y_c))
                        w_n = max(0,min(1,w_n)); h_n = max(0,min(1,h_n))
                        f.write(f"{ycls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                        class_counts[ycls] += 1
        except Exception as e:
            print("COCO processing failed:", e)
    else:
        print("COCO dataset not found -> skipping COCO stage.")

    # ---- Stats
    print("\n=== Dataset Stats ===")
    for i, cname in enumerate(NAMES):
        print(f"{cname}: {class_counts.get(i,0)} instances")
    if class_counts[2] == 0 or class_counts[3] == 0:
        print("âš ï¸� Note: No samples for reflex/polaroid cameras (ImageNet stage failed or not present).")

    # ---- Train/Val split
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(all_images)
    split = int(0.8 * len(all_images))
    train_images = all_images[:split]
    val_images   = all_images[split:]

    with open(os.path.join(DATASET_DIR, 'train.txt'), 'w') as f:
        for img in train_images:
            f.write(os.path.join(IMAGES_DIR, img) + '\n')
    with open(os.path.join(DATASET_DIR, 'val.txt'), 'w') as f:
        for img in val_images:
            f.write(os.path.join(IMAGES_DIR, img) + '\n')

    data_yaml = os.path.join(DATASET_DIR, 'data.yaml')
    with open(data_yaml, 'w') as f:
        f.write(f"train: {os.path.join(DATASET_DIR, 'train.txt')}\n")
        f.write(f"val: {os.path.join(DATASET_DIR, 'val.txt')}\n")
        f.write(f"nc: {NC}\n")
        f.write(f"names: {NAMES}\n")
    print(f"\nTotal images: {len(all_images)} | train: {len(train_images)} | val: {len(val_images)}")
    print(f"data.yaml -> {data_yaml}")

    return data_yaml, os.path.join(DATASET_DIR, 'train.txt'), os.path.join(DATASET_DIR, 'val.txt')


def autodetect_teacher_path(default_path):
    if os.path.exists(default_path):
        return default_path
    print(f"Teacher path not found: {default_path}. Scanning /kaggle/input for *.pt ...")
    pts = glob.glob('/kaggle/input/**/*.pt', recursive=True)
    if pts:
        print("Found candidate:", pts[0])
        return pts[0]
    raise FileNotFoundError("No teacher weights (.pt) found under /kaggle/input. Please set TEACHER_PATH correctly.")



def results_to_norm_lists(res, W, H):
    """Ultralytics Results -> XYXY normalized [0,1] lists for WBF."""
    if res.boxes is None or len(res.boxes) == 0:
        return [], [], []
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = [[x1/W, y1/H, x2/W, y2/H] for (x1,y1,x2,y2) in xyxy]
    return boxes, conf.tolist(), cls.tolist()

def xyxy_to_xywhn(boxes_xyxy):
    out = []
    for x1,y1,x2,y2 in boxes_xyxy:
        w = max(1e-6, x2-x1); h = max(1e-6, y2-y1)
        cx = x1 + w/2; cy = y1 + h/2
        out.append([cx, cy, w, h])
    return out

def photometric_augment(img, blur=None, brightness=None):
    im = img.copy()
    if blur and blur > 0:
        k = int(blur*2) | 1
        im = cv2.GaussianBlur(im, (k,k), blur)
    if brightness:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[...,2] = np.clip(hsv[...,2]*brightness, 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return im



class ServerFLKDTrainer:
    """
    Server-centric pipeline:
      - n client forward (student) on augmented views -> per-client detections
      - WBF across clients -> y_agg
      - teacher forward on same images -> y_T
      - WBF([y_T, y_agg], weights=[alpha, 1-alpha]) -> y_star
      - write pseudo labels (copy subset images) -> train student (1 epoch)
    """
    def __init__(self, teacher_path, student_variant='n', names=None, imgsz=640, out_dir='/kaggle/working/flkd'):
        self.names = names or ['person','phone','reflex_camera','polaroid_camera']
        self.nc = len(self.names)
        self.imgsz = imgsz
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        print("Loading teacher ...")
        self.teacher = YOLO(teacher_path)
        self.teacher.model.to(DEVICE).eval()

        print(f"Creating student ({student_variant}) ...")
        student_ckpts = {'n':'yolo11n.pt','s':'yolo11s.pt','m':'yolo11m.pt'}
        self.student = YOLO(student_ckpts[student_variant])
        self.student.model.to(DEVICE)

    def _client_forward_batch(self, rgbs, domain):
        aug_rgbs = [photometric_augment(im, domain.get('blur'), domain.get('brightness')) for im in rgbs]
        res = self.student(aug_rgbs, imgsz=self.imgsz, conf=0.001, iou=0.7, verbose=False)
        return res

    def _teacher_forward_batch(self, rgbs):
        res = self.teacher(rgbs, imgsz=self.imgsz, conf=0.001, iou=0.7, verbose=False)
        return res

    def _aggregate_clients_wbf(self, multi_client_results, sizes, weights=None,
                               iou_thr=0.55, skip_box_thr=0.001):
        C = len(multi_client_results)
        B = len(multi_client_results[0])
        fused = []
        for b in range(B):
            boxes_list, scores_list, labels_list = [], [], []
            H,W = sizes[b]
            for c in range(C):
                bxs, scs, lbs = results_to_norm_lists(multi_client_results[c][b], W, H)
                if len(bxs) > 0:  # <-- fixed truthiness check
                    boxes_list.append(bxs); scores_list.append(scs); labels_list.append(lbs)
            if len(boxes_list) > 0:  # <-- fixed
                ws = weights if (weights and len(weights)==len(boxes_list)) else None
                f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=ws, iou_thr=iou_thr, skip_box_thr=skip_box_thr
                )
            else:
                f_boxes, f_scores, f_labels = [], [], []
            fused.append({'boxes_norm': f_boxes, 'scores': f_scores, 'labels': f_labels})
        return fused

    def _mix_wbf(self, yT, yAgg, alpha=0.7, iou_thr=0.55):
        mixed = []
        for b in range(len(yT)):
            boxes_list, scores_list, labels_list = [], [], []
            if len(yT[b]['boxes_norm']) > 0:  # <-- fixed
                boxes_list.append(yT[b]['boxes_norm']); scores_list.append(yT[b]['scores']); labels_list.append(yT[b]['labels'])
            if len(yAgg[b]['boxes_norm']) > 0:  # <-- fixed
                boxes_list.append(yAgg[b]['boxes_norm']); scores_list.append(yAgg[b]['scores']); labels_list.append(yAgg[b]['labels'])
            if len(boxes_list) > 0:  # <-- fixed
                w = [alpha, 1.0-alpha][:len(boxes_list)]
                f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=w, iou_thr=iou_thr, skip_box_thr=0.001
                )
            else:
                f_boxes, f_scores, f_labels = [], [], []
            mixed.append({'boxes_norm': f_boxes, 'scores': f_scores, 'labels': f_labels})
        return mixed

    def _write_round_dataset(self, image_paths, fused_targets, save_root):
        """
        Copy selected images to save_root/images and write labels in save_root/labels
        so Ultralytics can find paired labels by replacing 'images' -> 'labels'.
        """
        img_dir = os.path.join(save_root, 'images'); os.makedirs(img_dir, exist_ok=True)
        lbl_dir = os.path.join(save_root, 'labels'); os.makedirs(lbl_dir, exist_ok=True)

        for p, tgt in zip(image_paths, fused_targets):
            # copy image
            dst_img = os.path.join(img_dir, os.path.basename(p))
            if not os.path.exists(dst_img):
                shutil.copy(p, dst_img)

            # write label
            stem = Path(p).stem
            lp = os.path.join(lbl_dir, f"{stem}.txt")
            keep = [(b,s,l) for b,s,l in zip(tgt['boxes_norm'], tgt['scores'], tgt['labels']) if s >= 0.25]
            if len(keep) == 0:
                open(lp, 'w').close()
                continue
            boxes_xywh = xyxy_to_xywhn([k[0] for k in keep])
            with open(lp, 'w') as f:
                for (cx,cy,w,h), (_,sc,lb) in zip(boxes_xywh, keep):
                    f.write(f"{int(lb)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # YAML
        data_yaml = os.path.join(save_root, 'data.yaml')
        with open(data_yaml, 'w') as f:
            f.write(f"path: {save_root}\n")
            f.write(f"train: images\n")
            f.write(f"val: images\n")
            f.write(f"nc: {self.nc}\n")
            f.write(f"names: {self.names}\n")
        return data_yaml

    def run(self, proxy_txt, num_rounds=3, num_clients=5, images_per_round=1200,
            alpha_start=1.0, alpha_end=0.5):
        with open(proxy_txt, 'r') as f:
            pool = [l.strip() for l in f.readlines()]
        print(f"[Server] Proxy pool size: {len(pool)}")
    
        domains = [{'blur': random.uniform(0.2, 1.5), 'brightness': random.uniform(0.7, 1.3)}
                   for _ in range(num_clients)]
    
        for r in range(num_rounds):
            alpha = alpha_start + (alpha_end - alpha_start) * (r / max(1, num_rounds-1))
            subset = random.sample(pool, min(images_per_round, len(pool)))
            print(f"\n=== Round {r+1}/{num_rounds} | alpha={alpha:.2f} | subset={len(subset)} ===")
    
            fused_all = []
            BATCH = 16
            for i in range(0, len(subset), BATCH):
                paths = subset[i:i+BATCH]
                rgbs, sizes = [], []
                for p in paths:
                    bgr = cv2.imread(p)
                    if bgr is None:  # skip silently
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgbs.append(rgb); sizes.append(rgb.shape[:2])
    
                if len(rgbs) == 0:
                    fused_all.extend([{'boxes_norm':[], 'scores':[], 'labels':[]}] * len(paths))
                    continue
    
                # client forwards
                per_client_results = []
                for c in range(num_clients):
                    res_c = self._client_forward_batch(rgbs, domains[c])
                    per_client_results.append(res_c)
    
                # WBF across clients
                y_agg = self._aggregate_clients_wbf(per_client_results, sizes)
    
                # teacher forward
                yT_raw = self._teacher_forward_batch(rgbs)
                y_T = []
                for b_idx, res in enumerate(yT_raw):
                    H, W = sizes[b_idx]
                    bxs, scs, lbs = results_to_norm_lists(res, W, H)
                    y_T.append({'boxes_norm': bxs, 'scores': scs, 'labels': lbs})
    
                # mix
                y_star = self._mix_wbf(y_T, y_agg, alpha=alpha)
                fused_all.extend(y_star)
    
            # write round dataset & train student (1 epoch)
            save_root = os.path.join(self.out_dir, f"round_{r+1}")
            os.makedirs(save_root, exist_ok=True)
            data_yaml = self._write_round_dataset(subset, fused_all, save_root)
    
            print(f"[Server] Train student on pseudo labels (round {r+1}) ...")
            self.student.train(
                data=data_yaml, imgsz=self.imgsz, epochs=1, batch=16,
                lr0=0.001, patience=0,
                device=(0 if DEVICE.type=='cuda' else 'cpu'),
                verbose=False, cache=False, save=True,
                project=self.out_dir, name=f"student_round_{r+1}"
            )
    
            # =========================
            # [FIX] Re-init student tá»« weight cá»§a vÃ²ng vá»«a train
            # Ä‘á»ƒ trÃ¡nh KeyError: 'model' khi gá»�i .train() á»Ÿ vÃ²ng tiáº¿p theo
            weights_dir = os.path.join(self.out_dir, f"student_round_{r+1}", "weights")
            resume_ckpt = None
            for nm in ("best.pt", "last.pt"):
                cand = os.path.join(weights_dir, nm)
                if os.path.exists(cand):
                    resume_ckpt = cand
                    break
            if resume_ckpt:
                print(f"[Server] Re-init student from {resume_ckpt}")
                self.student = YOLO(resume_ckpt)
                self.student.model.to(DEVICE)
            else:
                print(f"[Server][WARN] No weights in {weights_dir}; keep current student")
            # =========================
    
        print("\n[Server] FL+KD complete.")
        try:
            self.student.save(os.path.join(self.out_dir, 'student_flkd.pt'))
        except Exception:
            pass



def find_latest_student_ckpt(root=FLKD_OUTDIR):
    runs = sorted(glob.glob(os.path.join(root, "student_round_*")), key=lambda p: (len(p), p))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        return None
    last = runs[-1]
    for name in ["best.pt", "last.pt"]:
        cand = os.path.join(last, "weights", name)
        if os.path.exists(cand):
            return cand
    return None

def visualize_wbf(teacher_path, student_path, data_yaml_path, num_samples=6, alpha_wbf=0.7):
    teacher = YOLO(teacher_path)
    student = YOLO(student_path)

    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    val_file = cfg['val']
    if os.path.isdir(val_file):
        val_imgs = sorted([os.path.join(val_file, p) for p in os.listdir(val_file)])
    else:
        with open(val_file, 'r') as f:
            val_imgs = [l.strip() for l in f.readlines()]

    if len(val_imgs) == 0:
        print("No val images for visualization.")
        return

    samples = random.sample(val_imgs, min(num_samples, len(val_imgs)))
    fig, axes = plt.subplots(len(samples), 4, figsize=(22, 5*len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    for r, p in enumerate(samples):
        bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        t_res = teacher(p, verbose=False)[0]
        s_res = student(p, verbose=False)[0]
        t_img = t_res.plot(); s_img = s_res.plot()

        b1,s1,l1 = results_to_norm_lists(t_res, W, H)
        b2,s2,l2 = results_to_norm_lists(s_res, W, H)
        if (len(b1) > 0) or (len(b2) > 0):  # <-- fixed
            lists_b = [b1]; lists_s = [s1]; lists_l = [l1]; weights = [alpha_wbf]
            if len(b2) > 0:
                lists_b.append(b2); lists_s.append(s2); lists_l.append(l2); weights.append(1-alpha_wbf)
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                lists_b, lists_s, lists_l, weights=weights, iou_thr=0.55, skip_box_thr=0.001
            )
            fused = rgb.copy()
            for (x1,y1,x2,y2),sc,lb in zip(f_boxes,f_scores,f_labels):
                x1,y1,x2,y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
                cv2.rectangle(fused,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(fused, f"{NAMES[int(lb)]}:{sc:.2f}", (x1,max(0,y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            fused = rgb

        axes[r,0].imshow(rgb);   axes[r,0].set_title("Original"); axes[r,0].axis('off')
        axes[r,1].imshow(t_img); axes[r,1].set_title("Teacher");  axes[r,1].axis('off')
        axes[r,2].imshow(s_img); axes[r,2].set_title("Student");  axes[r,2].axis('off')
        axes[r,3].imshow(fused); axes[r,3].set_title("WBF(T+S)"); axes[r,3].axis('off')

    plt.tight_layout()
    out_png = '/kaggle/working/detection_comparison_wbf.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print("Saved visualization ->", out_png)
    plt.show()




def main():
    print("ğŸš€ Server-centric FL+KD (WBF) â€” End-to-End on Kaggle")
    data_yaml, train_txt, val_txt = prepare_dataset()
    teacher_path = autodetect_teacher_path(TEACHER_PATH)

    print("\nâœ… Dataset ready at:", data_yaml)
    print("âœ… Teacher weights:", teacher_path)

    server = ServerFLKDTrainer(
        teacher_path=teacher_path,
        student_variant=STUDENT_VARIANT,
        names=NAMES,
        imgsz=IMG_SIZE,
        out_dir=FLKD_OUTDIR
    )
    server.run(
        proxy_txt=train_txt,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        images_per_round=IMAGES_PER_ROUND,
        alpha_start=ALPHA_START,
        alpha_end=ALPHA_END
    )

    student_ckpt = find_latest_student_ckpt(FLKD_OUTDIR)
    if student_ckpt is not None:
        visualize_wbf(teacher_path, student_ckpt, data_yaml, num_samples=6, alpha_wbf=0.7)
    else:
        print("Student checkpoint not found for visualization.")



main()


# ==== CONFIG CHO KHá»�I Ä�Ã�NH GIÃ� ====
import os, glob, json, math, random, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ensemble_boxes import weighted_boxes_fusion

# Táº­n dá»¥ng cÃ¡c biáº¿n Ä‘Ã£ cÃ³ tá»« notebook cá»§a báº¡n:
# - DEVICE, NAMES, IMG_SIZE, DATASET_DIR, LABELS_DIR, FLKD_OUTDIR, TEACHER_PATH
# - cÃ¡c hÃ m autodetect_teacher_path(), find_latest_student_ckpt() Ä‘Ã£ Ä‘á»‹nh nghÄ©a á»Ÿ trÃªn

# ThÆ° má»¥c xuáº¥t hÃ¬nh
METRICS_DIR = '/kaggle/working/metrics_figs'
os.makedirs(METRICS_DIR, exist_ok=True)

# data.yaml Ä‘Ã£ Ä‘Æ°á»£c write trong prepare_dataset()
DATA_YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
assert os.path.exists(DATA_YAML_PATH), "KhÃ´ng tÃ¬m tháº¥y data.yaml; hÃ£y cháº¡y chuáº©n bá»‹ dataset trÆ°á»›c."

with open(DATA_YAML_PATH, 'r') as f:
    data_cfg = yaml.safe_load(f)

# Láº¥y danh sÃ¡ch áº£nh val
val_spec = data_cfg['val']
if os.path.isdir(val_spec):
    val_imgs = sorted([os.path.join(val_spec, p) for p in os.listdir(val_spec)
                       if p.lower().endswith(('.jpg','.jpeg','.png'))])
else:
    with open(val_spec, 'r') as f:
        val_imgs = [l.strip() for l in f.readlines() if l.strip()]

print(f"Sá»‘ áº£nh val: {len(val_imgs)}")

# Teacher & Student checkpoint
teacher_path = autodetect_teacher_path(TEACHER_PATH)
student_ckpt = find_latest_student_ckpt(FLKD_OUTDIR)
print("Teacher:", teacher_path)
print("Student (latest):", student_ckpt)

# Giá»›i háº¡n nhanh náº¿u cáº§n (Ä‘Ã¡nh giÃ¡ nhanh)
EVAL_LIMIT = None   # vÃ­ dá»¥ Ä‘áº·t 600 Ä‘á»ƒ cháº¡y nhanh hÆ¡n, hoáº·c None Ä‘á»ƒ dÃ¹ng toÃ n bá»™
if EVAL_LIMIT is not None:
    rng = np.random.RandomState(42)
    val_imgs = list(rng.choice(val_imgs, size=min(EVAL_LIMIT, len(val_imgs)), replace=False))
print("Ä�Ã¡nh giÃ¡ trÃªn:", len(val_imgs), "áº£nh")



def run_ultra_val(model_path, tag):
    model = YOLO(model_path)
    # project/name Ä‘á»ƒ gom hÃ¬nh vá»� thÆ° má»¥c metrics_figs/ultralytics/<tag>
    results = model.val(
        data=DATA_YAML_PATH,
        imgsz=IMG_SIZE,
        iou=0.55,           # khá»›p vá»›i WBF IoU báº¡n dÃ¹ng
        conf=0.001,
        plots=True,         # -> sinh PR, P/R/F1, confusion matrix, ...
        save_json=True,     # -> COCO-format JSON (náº¿u cáº§n)
        project=os.path.join(METRICS_DIR, 'ultralytics'),
        name=tag,
        verbose=False,
        device=(0 if torch.cuda.is_available() else 'cpu'),
    )
    outdir = os.path.join(METRICS_DIR, 'ultralytics', tag)
    print(f"[Ultralytics] Ä�Ã£ lÆ°u hÃ¬nh vÃ o: {outdir}")
    return results

_ = run_ultra_val(teacher_path, 'teacher')
if student_ckpt is not None:
    _ = run_ultra_val(student_ckpt, 'student')



def build_coco_gt_from_yolo(val_img_paths, labels_dir, class_names):
    images, annotations, img_id_map = [], [], {}
    ann_id = 1
    for idx, p in enumerate(val_img_paths):
        image_id = idx + 1
        img_id_map[p] = image_id
        # KÃ­ch thÆ°á»›c áº£nh
        try:
            with Image.open(p) as im:
                W, H = im.size
        except Exception:
            bgr = cv2.imread(p)
            if bgr is None:
                # bá»� qua áº£nh lá»—i
                continue
            H, W = bgr.shape[:2]

        images.append({
            'id': image_id,
            'file_name': os.path.basename(p),
            'width': W, 'height': H
        })

        stem = Path(p).stem
        # 2 á»©ng viÃªn Ä‘Æ°á»�ng dáº«n label: theo chuáº©n báº¡n Ä‘Ã£ ghi
        lp1 = os.path.join(labels_dir, f"{stem}.txt")
        lp2 = os.path.splitext(p.replace('/images/', '/labels/'))[0] + '.txt'
        lp = lp1 if os.path.exists(lp1) else (lp2 if os.path.exists(lp2) else None)

        if lp and os.path.exists(lp):
            with open(lp, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:5])
                    x = (cx - w/2) * W
                    y = (cy - h/2) * H
                    bw = w * W
                    bh = h * H
                    annotations.append({
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': c + 1,  # 1-based
                        'bbox': [float(max(0, x)), float(max(0, y)),
                                 float(max(1e-3, bw)), float(max(1e-3, bh))],
                        'area': float(max(1e-3, bw * bh)),
                        'iscrowd': 0
                    })
                    ann_id += 1

    categories = [{'id': i+1, 'name': name} for i, name in enumerate(class_names)]
    cocoGt = COCO()
    cocoGt.dataset = {'images': images, 'annotations': annotations, 'categories': categories}
    cocoGt.createIndex()
    return cocoGt, img_id_map

cocoGt, img_id_map = build_coco_gt_from_yolo(val_imgs, LABELS_DIR, NAMES)
print("COCO GT images:", len(cocoGt.dataset['images']),
      "| annos:", len(cocoGt.dataset['annotations']))



def predict_to_coco_dets(model_path, img_paths, img_id_map,
                         imgsz=640, conf=0.001, iou=0.55, batch=16):
    model = YOLO(model_path)
    # Ä‘áº©y model lÃªn GPU náº¿u cÃ³
    if hasattr(model, 'model'):
        model.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dets = []
    for i in range(0, len(img_paths), batch):
        batch_paths = img_paths[i:i+batch]
        results = model(batch_paths, imgsz=imgsz, conf=conf, iou=iou,
                        device=(0 if torch.cuda.is_available() else 'cpu'),
                        verbose=False)
        for p, res in zip(batch_paths, results):
            image_id = img_id_map[p]
            H, W = res.orig_shape
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss  = res.boxes.cls.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                # clip vÃ o áº£nh
                x1 = float(max(0.0, min(x1, W - 1)))
                y1 = float(max(0.0, min(y1, H - 1)))
                w  = float(max(1e-3, min(x2 - x1, W - 1 - x1)))
                h  = float(max(1e-3, min(y2 - y1, H - 1 - y1)))
                dets.append({
                    'image_id': image_id,
                    'category_id': int(c) + 1,
                    'bbox': [x1, y1, w, h],
                    'score': float(sc)
                })
    return dets

def eval_coco(cocoGt, dets):
    # [FIX] Má»™t sá»‘ báº£n pycocotools yÃªu cáº§u 'info' trong GT dataset
    if not hasattr(cocoGt, 'dataset') or 'info' not in cocoGt.dataset:
        cocoGt.dataset['info'] = {'description': 'GT built from YOLO labels', 'version': '1.0'}

    cocoDt = cocoGt.loadRes(dets)
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.params.iouThrs = np.linspace(0.5, 0.95, 10)
    E.params.maxDets = [100, 300, 300]

    # (khÃ´ng báº¯t buá»™c nhÆ°ng nÃªn cÃ³) Ä‘áº£m báº£o catIds/imgIds khá»›p
    E.params.catIds = [c['id'] for c in cocoGt.dataset['categories']]
    E.params.imgIds = [im['id'] for im in cocoGt.dataset['images']]

    E.evaluate(); E.accumulate(); E.summarize()

    stats = {
        'mAP_50_95': float(E.stats[0]),
        'mAP_50':    float(E.stats[1]),
        'mAP_75':    float(E.stats[2]),
        'AR_1':      float(E.stats[6]),
        'AR_10':     float(E.stats[7]),
        'AR_100':    float(E.stats[8]),
    }
    # AP theo lá»›p
    precisions = E.eval['precision']  # TxRxKxAxM
    ap_per_class = {}
    for idx, catId in enumerate(E.params.catIds):
        p = precisions[:, :, idx, 0, -1]
        p = p[p > -1]
        ap_per_class[catId] = float(np.mean(p)) if p.size else float('nan')
    return E, stats, ap_per_class


teacher_dets = predict_to_coco_dets(teacher_path, val_imgs, img_id_map, imgsz=IMG_SIZE, iou=0.55)
E_t, stats_t, apcls_t = eval_coco(cocoGt, teacher_dets)

if student_ckpt is not None:
    student_dets = predict_to_coco_dets(student_ckpt, val_imgs, img_id_map, imgsz=IMG_SIZE, iou=0.55)
    E_s, stats_s, apcls_s = eval_coco(cocoGt, student_dets)



def plot_ap_bar(ap_t, ap_s, class_names, out_path):
    x = np.arange(len(class_names))
    t = [ap_t.get(i+1, np.nan) for i in range(len(class_names))]
    s = [ap_s.get(i+1, np.nan) for i in range(len(class_names))]

    plt.figure(figsize=(10, 5))
    width = 0.38
    plt.bar(x - width/2, t, width, label='Teacher')
    plt.bar(x + width/2, s, width, label='Student')
    plt.xticks(x, class_names, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel('AP (0.5:0.95)')
    plt.title('Per-class AP â€” Teacher vs Student')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()

if student_ckpt is not None:
    out_png = os.path.join(METRICS_DIR, 'ap_per_class_teacher_vs_student.png')
    plot_ap_bar(apcls_t, apcls_s, NAMES, out_png)
    print("Saved:", out_png)



def plot_pr_overlay(E_t, E_s, class_names, iou=0.5, out_dir=METRICS_DIR):
    iou_thrs = E_t.params.iouThrs
    i = int(np.argmin(np.abs(iou_thrs - iou)))
    rec = E_t.params.recThrs
    for k, name in enumerate(class_names):
        p_t = E_t.eval['precision'][i, :, k, 0, -1]
        p_s = E_s.eval['precision'][i, :, k, 0, -1]
        p_t = np.where(p_t < 0, np.nan, p_t)
        p_s = np.where(p_s < 0, np.nan, p_s)

        plt.figure(figsize=(5.6, 5))
        plt.plot(rec, p_t, label='Teacher')
        plt.plot(rec, p_s, label='Student')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve (IoU={iou:.2f}) â€” {name}')
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        fn = os.path.join(out_dir, f'PR_{name}_IoU{iou:.2f}.png')
        plt.tight_layout()
        plt.savefig(fn, dpi=200, bbox_inches='tight')
        plt.show()
        print("Saved:", fn)

if student_ckpt is not None:
    plot_pr_overlay(E_t, E_s, NAMES, iou=0.5, out_dir=METRICS_DIR)



def find_round_ckpts(root=FLKD_OUTDIR):
    out = []
    for rdir in sorted(glob.glob(os.path.join(root, 'student_round_*'))):
        name = Path(rdir).name
        best = os.path.join(rdir, 'weights', 'best.pt')
        last = os.path.join(rdir, 'weights', 'last.pt')
        if os.path.exists(best):
            out.append((name, best))
        elif os.path.exists(last):
            out.append((name, last))
    return out

def map_over_rounds(round_ckpts, img_paths, img_id_map, cocoGt):
    rows = []
    for tag, ckpt in round_ckpts:
        dets = predict_to_coco_dets(ckpt, img_paths, img_id_map, imgsz=IMG_SIZE, iou=0.55)
        _, stats, _ = eval_coco(cocoGt, dets)
        rows.append({'round': tag, **stats})
    df = pd.DataFrame(rows)
    df['round_idx'] = df['round'].str.extract(r'(\d+)').astype(int)
    return df.sort_values('round_idx')

round_ckpts = find_round_ckpts(FLKD_OUTDIR)
if round_ckpts:
    df_round = map_over_rounds(round_ckpts, val_imgs, img_id_map, cocoGt)
    csv_path = os.path.join(METRICS_DIR, 'mAP_over_rounds.csv')
    df_round.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    # Váº½ mAP@[.5:.95] theo vÃ²ng
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df_round['round_idx'], df_round['mAP_50_95'], marker='o')
    plt.xlabel('Round')
    plt.ylabel('mAP (0.5:0.95)')
    plt.title('Student mAP vs. FL+KD Round')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(METRICS_DIR, 'mAP_vs_round.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved:", out_png)
else:
    print("KhÃ´ng tÃ¬m tháº¥y thÆ° má»¥c student_round_* trong", FLKD_OUTDIR)



def results_to_norm_lists(res, W, H):
    if res.boxes is None or len(res.boxes) == 0:
        return [], [], []
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
    boxes = [[x1/W, y1/H, x2/W, y2/H] for (x1,y1,x2,y2) in xyxy]
    return boxes, conf.tolist(), cls.tolist()

def fuse_ts_wbf_for_image(res_t, res_s, alpha, W, H, iou_thr=0.55):
    b1, s1, l1 = results_to_norm_lists(res_t, W, H)
    b2, s2, l2 = results_to_norm_lists(res_s, W, H)

    boxes_list, scores_list, labels_list, weights = [], [], [], []
    if len(b1) > 0:
        boxes_list.append(b1); scores_list.append(s1); labels_list.append(l1); weights.append(alpha)
    if len(b2) > 0:
        boxes_list.append(b2); scores_list.append(s2); labels_list.append(l2); weights.append(1 - alpha)

    if len(boxes_list) == 0:
        return []

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=0.001
    )
    dets = []
    for (x1n, y1n, x2n, y2n), sc, lb in zip(f_boxes, f_scores, f_labels):
        x1 = float(x1n * W); y1 = float(y1n * H)
        w  = float((x2n - x1n) * W); h = float((y2n - y1n) * H)
        dets.append({'bbox': [x1, y1, w, h], 'score': float(sc), 'category_id': int(lb) + 1})
    return dets

def eval_wbf_alpha(teacher_path, student_path, alpha_list, img_paths, img_id_map, cocoGt, batch=16):
    dets_per_alpha = {a: [] for a in alpha_list}
    mT = YOLO(teacher_path); mS = YOLO(student_path)
    if hasattr(mT, 'model'): mT.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if hasattr(mS, 'model'): mS.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    for i in range(0, len(img_paths), batch):
        batch_paths = img_paths[i:i+batch]
        res_t = mT(batch_paths, imgsz=IMG_SIZE, conf=0.001, iou=0.55,
                   device=(0 if torch.cuda.is_available() else 'cpu'), verbose=False)
        res_s = mS(batch_paths, imgsz=IMG_SIZE, conf=0.001, iou=0.55,
                   device=(0 if torch.cuda.is_available() else 'cpu'), verbose=False)
        for p, rt, rs in zip(batch_paths, res_t, res_s):
            H, W = rt.orig_shape
            for a in alpha_list:
                fused = fuse_ts_wbf_for_image(rt, rs, a, W, H, iou_thr=0.55)
                for d in fused:
                    det = d.copy()
                    det['image_id'] = img_id_map[p]
                    dets_per_alpha[a].append(det)

    rows = []
    for a in alpha_list:
        E, stats, _ = eval_coco(cocoGt, dets_per_alpha[a])
        rows.append({'alpha': float(a), **stats})
    df = pd.DataFrame(rows).sort_values('alpha', ascending=False)
    return df

if student_ckpt is not None:
    alphas = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    df_alpha = eval_wbf_alpha(teacher_path, student_ckpt, alphas, val_imgs, img_id_map, cocoGt, batch=16)
    csv_path = os.path.join(METRICS_DIR, 'mAP_vs_alpha_TplusS.csv')
    df_alpha.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df_alpha['alpha'], df_alpha['mAP_50_95'], marker='o')
    plt.xlabel('Î± (trá»�ng sá»‘ cho Teacher trong WBF)')
    plt.ylabel('mAP (0.5:0.95)')
    plt.title('WBF(T, S): mAP vs Î±')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    out_png = os.path.join(METRICS_DIR, 'mAP_vs_alpha_TplusS.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show()
    print("Saved:", out_png)



rows = [{'model': 'Teacher', **stats_t}]
if student_ckpt is not None:
    rows.append({'model': 'Student (final)', **stats_s})

summary_df = pd.DataFrame(rows)
summary_csv = os.path.join(METRICS_DIR, 'summary_metrics.csv')
summary_df.to_csv(summary_csv, index=False)
print("Summary CSV:", summary_csv)
summary_df


