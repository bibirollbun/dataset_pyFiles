!pip install ultralytics


import os
import cv2
import numpy as np
from ultralytics import YOLO
import joblib
from typing import List
from sklearn.metrics import f1_score
import pandas as pd

DATASET_ROOT = "/kaggle/input/cupybara/dataset/" 
EVAL_DIR = os.path.join(DATASET_ROOT, "test")

CLASS_NAMES = {
    0: 'armadillo',
    1: 'bird',
    2: 'capybara',
    3: 'cow',
    4: 'dusky_legged_guan',
    5: 'gray_brocket',
    6: 'hare',
    7: 'human',
    8: 'margay',
    9: 'skunk',
    10: 'unknown_animal',
    11: 'wild_boar'
}

def extract_16_frames(video_path: str) -> List[np.ndarray]:
    print(f"ğŸ”� Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=16, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        height, width = 480, 640
        frames = [np.zeros((height, width, 3), dtype=np.uint8)] * 16
    elif len(frames) < 16:
        frames += [frames[-1]] * (16 - len(frames))

    print(f"âœ… Extracted {len(frames)} frames")
    return frames

def aggregate_predictions(preds):
    print("ğŸ”� Aggregating predictions...")
    class_counts = {}
    for p in preds:
        for d in p.boxes.data.cpu().numpy():
            cls = int(d[5])
            name = CLASS_NAMES.get(cls)
            if name:
                class_counts[name] = class_counts.get(name, 0) + 1
    if not class_counts:
        print("âš ï¸� No detections found, returning 'no_animal'")
        return "'no_animal'"
    best_class = max(class_counts, key=class_counts.get)
    print(f"ğŸ�·ï¸� Final prediction: {best_class}")
    return f"'{best_class}'"

class BaseModel:
    def __init__(self):
        print("ğŸš€ Loading model...")
        self._load_model()
        print("âœ… Model loaded.")

    def _load_model(self) -> None:
        raise NotImplementedError

    def _predict(self, video_path: str) -> str:
        raise NotImplementedError

    def predict(self, video_path: str) -> str:
        return self._predict(video_path)

    def generate_submission(self, eval_dir: str) -> None:
        output_path = "submission.csv"
        filenames: List[str] = sorted(os.listdir(eval_dir))
        filenames = [f for f in filenames if f.endswith(".mp4")]  

        print(f"ğŸ“¼ Generating submission for {len(filenames)} videos...")

        submission = []
        for i, filename in enumerate(filenames):
            print(f"\nâ–¶ï¸� [{i+1}/{len(filenames)}] Processing {filename}")
            video_path = os.path.join(eval_dir, filename)
            prediction = self.predict(video_path)
            submission.append((filename.split(".")[0], prediction))

        submission_df = pd.DataFrame(submission, columns=["Filename", "Species"])
        submission_df.to_csv(output_path, index=False)
        print(f"\nâœ… Submission file saved to {output_path}")

class CustomModel(BaseModel):
    def _load_model(self) -> None:
        self.model = YOLO("/kaggle/input/baseline/baseline.pt")

    def _predict(self, video_path: str) -> str:
        frames = extract_16_frames(video_path)
        preds = [self.model.predict(frame, verbose=False)[0] for frame in frames]  # Predict frame by frame
        return aggregate_predictions(preds)



model = CustomModel()
model.generate_submission(eval_dir=EVAL_DIR)


df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv")).sort_values("Filename")
df_submission = pd.read_csv("submission.csv").sort_values("Filename")


f1_score(df_test["Species"].values, df_submission["Species"].values, average="weighted")

