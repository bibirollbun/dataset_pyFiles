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
#     video_path = "Video/1.mp4"   # replace with any video path
#     result = predict(video_path)
    
    
    
    
    
    
#     print(f"Prediction: {result}")



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

# video_path = "5.mp4"
# cap = cv2.VideoCapture(video_path)
# width, height = int(cap.get(3)), int(cap.get(4))
# ego_x, ego_y = width // 2, height   # ego car reference point

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
#         if len(trajectories[track_id]) >= 2:
#             (px, py) = trajectories[track_id][0]
#             (lx, ly) = trajectories[track_id][-1]
#             dx, dy = lx - px, ly - py

#             # Direction analysis
#             direction = ""
#             if abs(dx) > abs(dy):
#                 direction = "moving right" if dx > 0 else "moving left"
#             else:
#                 if dy > 0:
#                     direction = "approaching ego car"
#                 else:
#                     direction = "moving forward / away"

#             object_directions[name][track_id] = direction

# cap.release()

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



!pip install ultralytics 


import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from collections import defaultdict, deque
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 16
VIDEOS_DIR = "/kaggle/input/video-clips-of-accidents-for-traning-r3d-model/HEatmaps_full"
SUBMISSION_FILE = "submission.csv"
YOLO_FRAME_SKIP = 5   # process 1 in every 5 frames

# ============================================================
# Accident Classifier
# ============================================================
label_map = {0: "No Incident", 1: "Near Collision", 2: "Collision"}

class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.base(x)
        feats = feats.view(B, T, 512).mean(1)
        out = self.fc(feats)
        return out

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def sample_frames(frames, num_frames=NUM_FRAMES):
    if len(frames) == 0:
        return [torch.zeros(3,224,224)]*num_frames
    if len(frames) < num_frames:
        frames += [frames[-1]]*(num_frames - len(frames))
    idxs = torch.linspace(0, len(frames)-1, steps=num_frames, dtype=torch.long)
    return [frames[i] for i in idxs]

# Load accident model
accident_model = VideoClassifier().to(DEVICE)
accident_model.load_state_dict(torch.load(
    "/kaggle/input/1st-model_accident_classifire/pytorch/default/1/accident_model.pth",
    map_location=DEVICE
))
accident_model.eval()

def predict_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    frames = sample_frames(frames)
    frames = [transform(f) for f in frames]
    video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = accident_model(video_tensor)
        pred = out.argmax(1).item()

    return label_map[pred]

# ============================================================
# YOLO Detection + Object Tracking
# ============================================================
yolo_model = YOLO("yolov8n.pt").to(DEVICE)
ALLOWED_CLASSES = {"car", "truck", "bus", "person", "dog", "cat", "cow", "bicycle"}

def analyze_video(video_path):
    unique_objects = defaultdict(set)
    trajectories = defaultdict(lambda: deque(maxlen=5))
    object_directions = defaultdict(dict)
    incident_start_frame = None

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    ego_x, ego_y = width // 2, height
    cap.release()

    frame_idx = 0
    for result in yolo_model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",
        conf=0.6,
        verbose=False,
        show=False
    ):
        frame_idx += 1
        if frame_idx % YOLO_FRAME_SKIP != 0:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None
            name = yolo_model.names[cls]

            if conf < 0.6 or name not in ALLOWED_CLASSES or track_id is None:
                continue

            unique_objects[name].add(track_id)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trajectories[track_id].append((cx, cy))

            if len(trajectories[track_id]) >= 2:
                (px, py) = trajectories[track_id][0]
                (lx, ly) = trajectories[track_id][-1]
                dx, dy = lx - px, ly - py
                if abs(dx) > abs(dy):
                    direction = "moving right" if dx > 0 else "moving left"
                else:
                    direction = "approaching ego car" if dy > 0 else "moving forward / away"
                object_directions[name][track_id] = direction

                if incident_start_frame is None and direction == "approaching ego car":
                    incident_start_frame = frame_idx

    counts = {name: len(ids) for name, ids in unique_objects.items()}
    return counts, object_directions, incident_start_frame

# ============================================================
# Caption Generator
# ============================================================
def get_traffic_caption(counts):
    total_vehicles = counts.get("car",0)+counts.get("truck",0)+counts.get("bus",0)
    total_people = counts.get("person",0)
    total_animals = counts.get("dog",0)+counts.get("cat",0)+counts.get("cow",0)

    if total_vehicles == 0:
        traffic = "empty road"
    elif total_vehicles <= 3:
        traffic = "light traffic"
    elif total_vehicles <= 6:
        traffic = "moderate traffic"
    else:
        traffic = "heavy traffic"

    caption = f"Ego-car is driving on {traffic}."
    if total_people > 0:
        caption += f" There are {total_people} pedestrians nearby."
    if total_animals > 0:
        caption += " Some animals are on or near the road."
    return caption

# ============================================================
# Crash Severity & Recognition Mapping
# ============================================================
def map_incident_fields(incident, counts, object_directions, start_frame):
    # No incident case
    if incident == "No Incident":
        return {
            "Incident window start frame": -1,
            "Incident Detection": -1,
            "Crash Severity": "0. No Crash",
            "Ego-car involved": 0,
            "Incident Recognition": "unknown",
            "Caption Before Incident": "no incident",
            "Reason of Incident": "no incident"
        }

    # Hazard / Accident mapping
    if incident == "Near Collision":
        detection = 0  # Hazard
    else:
        detection = 1  # Accident

    # Ego involvement
    ego_involved = 1 if any(
        "approaching ego car" in d for obj in object_directions.values() for d in obj.values()
    ) else 0

    # Crash severity
    if detection == 0:
        crash_severity = "6. One or Multiple vehicles collided but ego-car is fine"
    else:
        crash_severity = "5. Multiple vehicles collided with ego-car" if ego_involved else "4. Other cars collided with person/car/object but ego-car is ok"

    # Incident recognition heuristic
    if counts.get("person",0) > 0 and ego_involved:
        recognition = "ego-car hits a pedestrian"
    elif counts.get("bicycle",0) > 0 and ego_involved:
        recognition = "ego-car hits a crossing cyclist"
    elif counts.get("dog",0)+counts.get("cat",0)+counts.get("cow",0) > 0 and ego_involved:
        recognition = "ego-car hit an animal"
    elif ego_involved:
        recognition = "ego-car hits a vehicle"
    else:
        recognition = "vehicle drives into another vehicle"

    caption = get_traffic_caption(counts)
    reason = f"{recognition.replace('_',' ')} detected."

    return {
        "Incident window start frame": start_frame if start_frame else 1,
        "Incident Detection": detection,
        "Crash Severity": crash_severity,
        "Ego-car involved": ego_involved,
        "Incident Recognition": recognition,
        "Caption Before Incident": caption,
        "Reason of Incident": reason
    }

# ============================================================
# Process Videos
# ============================================================
rows = []
videos = [v for v in os.listdir(VIDEOS_DIR) if v.endswith(".mp4")]

for idx, vid in enumerate(tqdm(videos, desc="Processing videos")):
    video_path = os.path.join(VIDEOS_DIR, vid)

    incident = predict_accident(video_path)
    counts, object_directions, start_frame = analyze_video(video_path)
    incident_fields = map_incident_fields(incident, counts, object_directions, start_frame)

    row = {
        "video": idx,
        **incident_fields,
        "Number of Bicyclists/Scooters": counts.get("bicycle",0),
        "Number of animals involved": counts.get("dog",0)+counts.get("cat",0)+counts.get("cow",0),
        "Number of pedestrians involved": counts.get("person",0),
        "Number of vehicles involved (excluding ego-car)": counts.get("car",0)+counts.get("truck",0)+counts.get("bus",0),
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(SUBMISSION_FILE, index=False)
print(f"âœ… Submission file saved to {SUBMISSION_FILE}")


