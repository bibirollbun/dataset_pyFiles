import os
import random
import json
import joblib
import shutil
import pandas as pd
import numpy as np
from typing import List

import torch
import cv2
from torchvision.models.video import r2plus1d_18
from torchvision import transforms

from sklearn.metrics import f1_score


# Hard coded class mappings
idx_to_class = {
    0: 'armadillo',
    1: 'bird',
    2: 'capybara',
    3: 'cow',
    4: 'dusky_legged_guan',
    5: 'gray_brocket',
    6: 'hare',
    7: 'human',
    8: 'insect',
    9: 'margay',
    10: 'no_animal',
    11: 'skunk',
    12: 'unknown_animal',
    13: 'wild_boar'
}

class_to_idx = {v: k for k, v in idx_to_class.items()}
LABELS = [f"'{idx_to_class[i]}'" for i in sorted(idx_to_class)]

print("Labels:", LABELS)


!ls /kaggle/input/cupybara/dataset/


DATASET_ROOT = "/kaggle/input/cupybara/dataset/dataset/"
EVAL_DIR =  os.path.join(DATASET_ROOT,"test")  # Directory containing evaluation `.mp4` files
TRAIN = True

# Video model configuration
MODEL_PATH = "/kaggle/input/cupybara-model-r2plus-aug-combined-es/trained_video_model_r2plus_aug_combined_equal_spacing.pt"
VIDEO_FRAMES = 16
FRAME_SIZE = 112  # Model input requirement
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


!rm -rf model_weights/ model_weights.zip submission.csv
!mkdir -p model_weights/
!cp /kaggle/input/cupybara-model-r2plus-aug-combined-es/trained_video_model_r2plus_aug_combined_equal_spacing.pt model_weights/
!zip -r model_weights.zip model_weights/


# Video model transforms and functions
transform = transforms.Compose([
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645],
                         [0.22803, 0.22145, 0.216989]),
])

def load_video_tensor(path, num_frames=VIDEO_FRAMES):
    """Load video as tensor with proper frame sampling"""
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idxs = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = transforms.ToPILImage()(frame)
            frames.append(transform(pil_image))

    cap.release()

    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))  # pad with last frame if short

    video_tensor = torch.stack(frames)  # [T, C, H, W]
    return video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

def load_model():
    """Load the trained video model"""
    model = r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

if TRAIN: 
    print("Video model setup complete. Training mode is enabled but not needed for submission.")


!ls /kaggle/input/native-fauna-dummy-model


class BaseModel:
    def __init__(self):
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError("You must implement `_load_model`.")

    def _predict(self, video_path: str) -> str:
        raise NotImplementedError("You must implement `_predict`.")

    def predict(self, video_path: str) -> str:
        return LABELS[self._predict(video_path)]

    def generate_submission(self, eval_dir: str) -> None:

        output_path: str = "submission.csv"
        filenames: List[str] = sorted(os.listdir(eval_dir))
        submission = []
        for filename in filenames:
            if filename.endswith(".mp4"):
                video_path = os.path.join(eval_dir, filename)
                prediction = self.predict(video_path)
                submission.append((filename.split(".")[0], f"{prediction}"))  # Label in single quotes
        
        submission_df = pd.DataFrame(submission, columns=["Filename", "Species"])
        submission_df.to_csv(output_path, index=False)
        print(f"âœ… Submission file saved to {output_path}")


class CustomModel(BaseModel):

    def _load_model(self) -> None:
        """
        Loads the video model from the model weights directory.
        This guarantees that the model runs OFFLINE
        """
        # Check if we're in the Kaggle environment with the uploaded dataset
        if os.path.exists("/kaggle/input/native-fauna-dummy-model"):
            model_path = "/kaggle/input/native-fauna-dummy-model/trained_video_model_r2plus_insects_aug_newset4.pt"
        else:
            # Local testing environment
            model_path = MODEL_PATH
            
        self.model = r2plus1d_18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_to_idx))
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.model = self.model.to(DEVICE)
        print(f"âœ… Model loaded from {model_path}")

    def _predict(self, video_path: str) -> int:
        """
        Uses the loaded video model to generate a prediction.
        """
        video = load_video_tensor(video_path).unsqueeze(0).to(DEVICE)  # [1, C, T, H, W]
        with torch.no_grad():
            outputs = self.model(video)
            pred_idx = outputs.argmax(dim=1).item()
        return pred_idx


model = CustomModel()
model.generate_submission(eval_dir=EVAL_DIR)


df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv")).sort_values("Filename")
df_submission = pd.read_csv("submission.csv").sort_values("Filename")


f1_score(df_test["Species"].values, df_submission["Species"].values, average="weighted")

