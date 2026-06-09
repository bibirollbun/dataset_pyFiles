# #--------- First we have to import our libraries ---------
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision
# import torchvision.transforms as transforms
# import os, cv2, random
# import numpy as np

# # --------- then we have to do some configrations ---------
# ACCIDENT_PATH = "/kaggle/input/accident-data-sets-cnn-vary-small/Accident"
# NO_INCIDENT_PATH = "/kaggle/input/accident-data-sets-cnn-vary-small/NO incident"
# NUM_FRAMES = 16   
# IMG_SIZE = 112
# BATCH_SIZE = 4
# EPOCHS = 12
# LR = 1e-4


# # --------- and next you we have to load and some prossesing of the data ---------
# class AccidentVideoDataset(Dataset):
#     def __init__(self, accident_dir, no_incident_dir, transform=None, num_frames=16):
#         self.samples = []
#         self.transform = transform
#         self.num_frames = num_frames

#         for f in os.listdir(accident_dir):
#             self.samples.append((os.path.join(accident_dir, f), 1))
#         for f in os.listdir(no_incident_dir):
#             self.samples.append((os.path.join(no_incident_dir, f), 0))

#     def __len__(self):
#         return len(self.samples)

#     def _load_video(self, path):
#         cap = cv2.VideoCapture(path)
#         frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#             frames.append(frame)
#         cap.release()

#         # Sample fixed number of frames
#         if len(frames) >= self.num_frames:
#             idxs = np.linspace(0, len(frames)-1, self.num_frames).astype(int)
#             frames = [frames[i] for i in idxs]
#         else:  # pad with last frame
#             while len(frames) < self.num_frames:
#                 frames.append(frames[-1])

#         frames = np.stack(frames)  # (T, H, W, C)
#         frames = torch.from_numpy(frames).permute(3,0,1,2)  # (C,T,H,W)
#         frames = frames.float() / 255.0
#         return frames

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         video = self._load_video(path)
#         if self.transform:
#             video = self.transform(video)
#         return video, label

# # --------- choose the model we have to use  ---------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torchvision.models.video.r3d_18(weights="KINETICS400_V1")
# model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
# model = model.to(device)

# # --------- then the treaning of the model ---------
# dataset = AccidentVideoDataset(ACCIDENT_PATH, NO_INCIDENT_PATH, num_frames=NUM_FRAMES)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for videos, labels in dataloader:
#         videos, labels = videos.to(device), torch.tensor(labels).to(device)
#         optimizer.zero_grad()
#         outputs = model(videos)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

# # --------- finally we have to save the model ---------
# torch.save(model.state_dict(), "accident_classifier.pth")
# print("✅ Model saved as accident_classifier.pth")




# import cv2
# from ultralytics import YOLO
# from collections import defaultdict, deque
# import torch

# # ------------------------------
# # Config
# # ------------------------------
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = YOLO("yolov8m.pt").to(device)

# ALLOWED_CLASSES = {"car", "truck", "bus", "person", "dog", "cat", "cow"}

# # Tracking
# unique_objects = defaultdict(set)
# trajectories = defaultdict(lambda: deque(maxlen=5))
# object_directions = defaultdict(dict)

# video_path = "Video/525.mp4"
# cap = cv2.VideoCapture(video_path)
# width, height = int(cap.get(3)), int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

# ego_x, ego_y = width // 2, height   # ego car reference point

# # Output video writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# # ------------------------------
# # Process video
# # ------------------------------
# for result in model.track(
#     source=video_path,
#     stream=True,
#     tracker="bytetrack.yaml",
#     conf=0.6,   # stricter confidence
#     verbose=False,
#     show=False
# ):

#     frame = result.orig_img  # get the current frame

#     for box in result.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])
#         track_id = int(box.id[0]) if box.id is not None else None
#         name = model.names[cls]

#         if conf < 0.6 or name not in ALLOWED_CLASSES or track_id is None:
#             continue

#         unique_objects[name].add(track_id)

#         # Center point of detection
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

#         # Save trajectory
#         trajectories[track_id].append((cx, cy))
#         direction = ""

#         if len(trajectories[track_id]) >= 2:
#             (px, py) = trajectories[track_id][0]
#             (lx, ly) = trajectories[track_id][-1]
#             dx, dy = lx - px, ly - py

#             # Direction analysis
#             if abs(dx) > abs(dy):
#                 direction = "→ right" if dx > 0 else "← left"
#             else:
#                 if dy > 0:
#                     direction = "↓ approaching"
#                 else:
#                     direction = "↑ moving away"

#             object_directions[name][track_id] = direction

#         # ----------------- Draw on frame -----------------
#         color = (0, 255, 0) if name in ["car", "truck", "bus"] else (255, 0, 0)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         label = f"{name} #{track_id} {direction}"
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

#     # Write annotated frame to output video
#     out.write(frame)

# cap.release()
# out.release()

# # ------------------------------
# # Final Summary
# # ------------------------------
# print("\nFinal Video Summary:")
# for name, ids in unique_objects.items():
#     print(f"{name}: {len(ids)}")

# # Aggregate movement summary
# summary_directions = defaultdict(lambda: defaultdict(int))
# for name, objs in object_directions.items():
#     for _, direction in objs.items():
#         summary_directions[name][direction] += 1

# print("\nTraffic Narrative:")
# print("Ego car is driving in traffic.")

# for name, count in unique_objects.items():
#     n = len(count)
#     if n == 0:
#         continue

#     directions = summary_directions[name]
#     if directions:
#         dir_text = ", ".join([f"{v} {k}" for k, v in directions.items()])
#         print(f"Detected {n} {name}(s), with movements: {dir_text}.")
#     else:
#         print(f"Detected {n} {name}(s).")




# import os
# import cv2
# import torch
# import torch.nn as nn
# from torchvision import models, transforms

# # ------------------------
# # Config
# # ------------------------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_FRAMES = 16
# MODEL_PATH = "accident_model.pth"  # make sure this is in same folder

# # ------------------------
# # Label map
# # ------------------------
# label_map = {0: "No Incident", 1: "Near Collision", 2: "Collision"}
# # ------------------------
# # Model Definition
# # ------------------------
# class VideoClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         base.fc = nn.Identity()
#         self.base = base
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):  # x: [B,T,C,H,W]
#         B, T, C, H, W = x.shape
#         x = x.view(B*T, C, H, W)
#         feats = self.base(x)
#         feats = feats.view(B, T, 512).mean(1)
#         out = self.fc(feats)
#         return out

# # ------------------------
# # Frame preprocessing
# # ------------------------
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std=[0.229,0.224,0.225])
# ])

# def sample_frames(frames, num_frames=NUM_FRAMES):
#     if len(frames) == 0:
#         return [torch.zeros(3,224,224)]*num_frames
#     if len(frames) < num_frames:
#         frames += [frames[-1]]*(num_frames - len(frames))
#     idxs = torch.linspace(0, len(frames)-1, steps=num_frames, dtype=torch.long)
#     return [frames[i] for i in idxs]

# # ------------------------
# # Load Model
# # ------------------------
# model = VideoClassifier().to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()

# # ------------------------
# # Predict Function
# # ------------------------
# def predict(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#     cap.release()

#     frames = sample_frames(frames)
#     frames = [transform(f) for f in frames]
#     video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         out = model(video_tensor)
#         pred = out.argmax(1).item()

#     return label_map[pred]

# # ------------------------
# # Example usage
# # ------------------------
# if __name__ == "__main__":
#     video_path = "Video/525.mp4"   # replace with any video path
#     result = predict(video_path)
#     print(f"Prediction: {result}")



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
NUM_FRAMES_ACCIDENT = 32 # in this code i have chage it from 24 to 32 best is 24 i was just cheaking if it work on 32 more best 
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

