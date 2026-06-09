!pip install ultralytics


import os
import cv2
import csv
import math
import logging
from collections import defaultdict, deque

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES_ACCIDENT = 32
ACCIDENT_MODEL_PATH = "/kaggle/input/1st-model_accident_classifire/pytorch/default/1/accident_model.pth"
VIDEO_DIR = "/kaggle/input/video-clips-of-accidents-for-traning-r3d-model/HEatmaps_full"
OUTPUT_CSV = "submission.csv"
ALLOWED_CLASSES = {"car", "truck", "bus", "person", "dog", "cat", "cow", "bicycle"}

# YOLO config
YOLO_WEIGHTS = "yolov8m.pt"
YOLO_CONF = 0.6
YOLO_TRACKER = "bytetrack.yaml"  # keep user supplied

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------------------------
# Label maps
# ------------------------------
incident_label_map = {0: "No incident", 1: "Near Collision", 2: "Collision"}
crash_severity_map = {
    0: "0. No Crash",
    1: "3. Possible crash, low severity",
    2: "4. Other cars collided with person/car/object but ego-car is ok"
}
incident_type_map = {
    0: "no incident",
    1: "near miss with another vehicle",
    2: "vehicle drives into another vehicle"
}

# ------------------------------
# Accident Classifier (Video -> label)
# ------------------------------
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.base(x)  # [B*T, 512]
        feats = feats.view(B, T, -1).mean(1)  # [B, 512]
        out = self.fc(feats)
        return out

# transform for each frame (keeps same input size as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def sample_frames(frames, num_frames=NUM_FRAMES_ACCIDENT):
    """
    Deterministic temporal sampling with edge padding.
    frames: list of HxWx3 uint8 (RGB)
    Returns a list of length num_frames of numpy arrays (still in uint8) suitable for transform.
    """
    n = len(frames)
    if n == 0:
        # return black frames
        blank = (255 * np.zeros((224, 224, 3), dtype='uint8')).astype('uint8')
        return [blank for _ in range(num_frames)]

    if n >= num_frames:
        # evenly spaced indices
        idxs = [int(round(i)) for i in torch.linspace(0, n - 1, steps=num_frames).tolist()]
        return [frames[i] for i in idxs]
    else:
        # pad by repeating last frame
        padded = frames + [frames[-1]] * (num_frames - n)
        idxs = [int(round(i)) for i in torch.linspace(0, len(padded) - 1, steps=num_frames).tolist()]
        return [padded[i] for i in idxs]

# Safe model load with checks
def load_accident_model(path, device):
    model = VideoClassifier().to(device)
    if not os.path.exists(path):
        logging.warning("Accident model path does not exist: %s. Returning uninitialized model.", path)
        return model
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
        model.eval()
        logging.info("Accident classifier loaded from %s", path)
    except Exception as e:
        logging.exception("Failed to load state dict. Returning model in eval mode anyway.")
        model.eval()
    return model

# batch inference helper (in case we want to classify multiple windows later)
def classify_incident_frames(model, frames, device):
    """
    frames: list of RGB frames (uint8 arrays) of arbitrary length.
    returns predicted label int.
    """
    sampled = sample_frames(frames)
    tensor_frames = [transform(f) for f in sampled]
    video_tensor = torch.stack(tensor_frames).unsqueeze(0).to(device)  # [1, T, C, H, W]
    with torch.no_grad():
        out = model(video_tensor)
        pred = int(out.argmax(1).item())
    return pred

# ------------------------------
# Incident frame finder (improved heuristic)
# ------------------------------
import numpy as np

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def find_incident_frame(yolo_results, distance_threshold=100, approach_threshold=5.0):
    """
    Improved heuristic:
      - For tracked objects across consecutive frames we compute approximate speeds (pixel/frame)
      - If a pair of objects' centers approach each other by more than approach_threshold when within distance_threshold,
        mark the frame where approach peaks as candidate incident frame.
    yolo_results: list of result objects (in temporal order). Each result must have .boxes.xyxy (numpy) and .boxes.id.
    """
    if not yolo_results:
        return 0

    # Build map: frame_idx -> list of (track_id, center)
    frames_centers = []
    for res in yolo_results:
        centers = []
        if not res.boxes:
            frames_centers.append(centers)
            continue
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
        ids = []
        # Try to get ids; ultralytics stores it differently across versions
        try:
            ids_arr = [int(x) for x in res.boxes.id.cpu().numpy()]
        except Exception:
            ids_arr = [None] * len(boxes)
        for b, tid in zip(boxes, ids_arr):
            centers.append((tid, box_center(b)))
        frames_centers.append(centers)

    best_frame = len(yolo_results) // 2
    best_score = 0.0

    # For every pair of consecutive frames, compute approach for matched ids
    # We build small map of id -> previous center to compute velocities
    prev_centers = {}
    for t, centers in enumerate(frames_centers):
        # current map id -> center
        cur_map = {tid: c for tid, c in centers if tid is not None}

        # compute pairwise distances among current centers and compare to previous frame's distances
        ids = list(cur_map.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                c_a = cur_map[id_a]; c_b = cur_map[id_b]
                dist_cur = euclid(c_a, c_b)
                # try previous
                if id_a in prev_centers and id_b in prev_centers:
                    dist_prev = euclid(prev_centers[id_a], prev_centers[id_b])
                    # approaching if distance decreased significantly
                    approach = dist_prev - dist_cur
                    if dist_cur < distance_threshold and approach > approach_threshold:
                        # score more if approach is larger and current proximity is small
                        score = approach * (distance_threshold - dist_cur) / (distance_threshold + 1e-6)
                        if score > best_score:
                            best_score = score
                            best_frame = t

        # update prev_centers for next iteration
        for tid, c in cur_map.items():
            prev_centers[tid] = c

    return int(best_frame)

# ------------------------------
# YOLO Model load (single global instance)
# ------------------------------
def load_yolo(weights, device):
    try:
        yolo = YOLO(weights).to(device)
        logging.info("YOLO loaded with weights: %s", weights)
        return yolo
    except Exception:
        logging.exception("Failed to load YOLO model. Check weights path.")
        raise

# ------------------------------
# Caption generator
# ------------------------------
def traffic_caption(counts):
    if counts["vehicles"] >= 10:
        return "Ego-car is driving in heavy traffic."
    elif 5 <= counts["vehicles"] <= 9:
        return "Ego-car is driving with moderate traffic."
    elif 1 <= counts["vehicles"] <= 4:
        return "Ego-car is driving with light traffic."
    else:
        return "Ego-car is driving on an empty road."

# ------------------------------
# Utilities
# ------------------------------
def safe_union(*sets):
    result = set()
    for s in sets:
        result |= (s if isinstance(s, set) else set())
    return result

# ------------------------------
# Main processing
# ------------------------------
def process_videos(video_dir, output_csv):
    accident_model = load_accident_model(ACCIDENT_MODEL_PATH, DEVICE)
    yolo = load_yolo(YOLO_WEIGHTS, DEVICE)

    videos = [v for v in os.listdir(video_dir) if v.lower().endswith(".mp4")]
    logging.info("Found %d videos in %s", len(videos), video_dir)

    # Ensure header
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video","Incident window start frame","Incident Detection","Crash Severity",
            "Ego-car involved","Label","Number of Bicyclists/Scooters",
            "Number of animals involved","Number of pedestrians involved",
            "Number of vehicles involved (excluding ego-car)",
            "Caption Before Incident","Reason of Incident"
        ])

        for vid in tqdm(videos, desc="Processing videos"):
            video_path = os.path.join(video_dir, vid)
            unique_objects = defaultdict(set)
            all_frames = []
            all_results = []

            try:
                # Use YOLO tracking (stream mode)
                for result in yolo.track(source=video_path, stream=True,
                                         tracker=YOLO_TRACKER, conf=YOLO_CONF, verbose=False, show=False):
                    frame = result.orig_img
                    # convert to RGB for classifier transform
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    all_frames.append(frame_rgb)
                    all_results.append(result)

                    # loop boxes safely
                    try:
                        boxes = list(result.boxes)
                    except Exception:
                        boxes = []
                    for box in boxes:
                        # robust extraction for cls/id/conf fields
                        try:
                            cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
                            conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
                        except Exception:
                            continue
                        try:
                            track_id = int(box.id[0]) if (hasattr(box, "id") and box.id is not None) else None
                        except Exception:
                            track_id = None

                        name = yolo.names.get(cls, str(cls)) if hasattr(yolo, "names") else str(cls)

                        if conf < YOLO_CONF or name not in ALLOWED_CLASSES or track_id is None:
                            continue
                        unique_objects[name].add(track_id)
            except Exception:
                logging.exception("YOLO tracking failed for video %s. Attempting fallback frame read.", video_path)
                # fallback: read video frames directly (no tracking)
                cap = cv2.VideoCapture(video_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    all_frames.append(frame_rgb)
                cap.release()

            # classify incident (safe)
            try:
                pred = classify_incident_frames(accident_model, all_frames, DEVICE)
            except Exception:
                logging.exception("Accident classifier failed; marking as 'No incident'")
                pred = 0

            # dynamic incident frame only if incident
            incident_window = find_incident_frame(all_results) if pred != 0 else 0

            # Aggregate counts
            vehicle_count = len(safe_union(unique_objects["car"], unique_objects["truck"], unique_objects["bus"]))
            vehicle_count = max(vehicle_count, 1)  # include ego-car as minimum 1
            counts = {
                "vehicles": vehicle_count,
                "pedestrians": len(unique_objects["person"]),
                "animals": len(safe_union(unique_objects["dog"], unique_objects["cat"], unique_objects["cow"])),
                "bicycles": len(unique_objects["bicycle"])
            }

            ego_involved = 1 if pred == 2 else 0

            before_caption = traffic_caption(counts)
            after_caption_map = {
                0: "No accident occurred.",
                1: "Other vehicles collided near ego-car.",
                2: "A collision involving another vehicle was detected."
            }
            after_caption = after_caption_map.get(pred, "No accident occurred.")

            writer.writerow([
                vid.replace(".mp4", ""),
                incident_window,
                incident_label_map.get(pred, "No incident"),
                crash_severity_map.get(pred, "0. No Crash"),
                ego_involved,
                incident_type_map.get(pred, "no incident"),
                counts["bicycles"], counts["animals"], counts["pedestrians"], counts["vehicles"],
                before_caption, after_caption
            ])

    logging.info("✅ Processing done. CSV saved at %s", output_csv)


# ------------------------------
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    # Validate input paths
    if not os.path.isdir(VIDEO_DIR):
        logging.error("VIDEO_DIR does not exist: %s", VIDEO_DIR)
    else:
        process_videos(VIDEO_DIR, OUTPUT_CSV)


