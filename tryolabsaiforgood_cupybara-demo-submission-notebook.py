import os
import random
import json
import joblib
import shutil
import pandas as pd
import numpy as np
from typing import List

from sklearn.metrics import f1_score


LABELS = ["'armadillo'", "'bird'", "'capybara'", "'cow'", "'dusky_legged_guan'",
    "'gray_brocket'", "'hare'", "'human'", "'insect'", "'margay'", "'no_animal'",
    "'skunk'", "'unknown_animal'", "'wild_boar'"]


!ls /kaggle/input/cupybara/dataset/


DATASET_ROOT = "/kaggle/input/cupybara/dataset/"
EVAL_DIR =  os.path.join(DATASET_ROOT,"test")  # Directory containing evaluation `.mp4` files
TRAIN = True


!rm -rf model_weights/ model_weights.zip submission.csv


from sklearn.ensemble import RandomForestClassifier

def extract_feature_from_filename(filename, max_len=16):
    """
    Convert a filename string into a fixed-length vector of ASCII codes.
    Pads with zeros if filename is shorter than max_len.
    """
    ascii_vals = [ord(c) for c in filename if c.isalnum()]
    ascii_vals = ascii_vals[:max_len]  # Truncate if too long
    return ascii_vals + [0] * (max_len - len(ascii_vals))


if TRAIN: 

    df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv"))
    label_to_index = {label: idx for idx, label in enumerate(LABELS)}
    
    
    X_test = np.array([extract_feature_from_filename(f) for f in df_test["Filename"]])
    y_test = df_test["Species"].map(label_to_index)

    # Create fake training set using filenames from test
    X_train = X_test
    y_train = y_test

    model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    shutil.rmtree("model_weights/", ignore_errors=True)
    os.makedirs("model_weights", exist_ok=True)
    joblib.dump(model, "model_weights/model.joblib")
    shutil.make_archive("model_weights", 'zip', "model_weights")


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
        Loads the model from a public dataset previously uploaded.
        This guarantees that the model runs OFFLINE
        """
        self.model = joblib.load("/kaggle/input/native-fauna-dummy-model/model.joblib")

    def _predict(self, video_path: str) -> int:
        """
        Uses the loaded model to generate a prediction.
        """
        filename = os.path.basename(video_path).split(".")[0]
        feature = extract_feature_from_filename(filename)
        pred = self.model.predict([feature])
        return pred[0]


model = CustomModel()
model.generate_submission(eval_dir=EVAL_DIR)


df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv")).sort_values("Filename")
df_submission = pd.read_csv("submission.csv").sort_values("Filename")


f1_score(df_test["Species"].values, df_submission["Species"].values, average="weighted")

