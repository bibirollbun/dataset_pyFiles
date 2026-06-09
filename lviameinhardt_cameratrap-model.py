import os
import random
import json
import joblib
import shutil
import pandas as pd
import numpy as np
from typing import List

import kagglehub
from speciesnet import SpeciesNet,DEFAULT_MODEL
import cv2
from multiprocessing import Pool

from sklearn.metrics import f1_score


LABELS = ["'armadillo'", "'bird'", "'capybara'", "'cow'", "'dusky_legged_guan'",
    "'gray_brocket'", "'hare'", "'human'", "'insect'", "'margay'", "'no_animal'",
    "'skunk'", "'unknown_animal'", "'wild_boar'"]


DATASET_ROOT = "/kaggle/input/cupybara/dataset/dataset"
EVAL_DIR =  os.path.join(DATASET_ROOT,"test")  # Directory containing evaluation `.mp4` files
TRAIN = False


def extract_spaced_frames_fast(video_info):

    video_path, output_folder, max_frames = video_info
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_folder, video_name)

    if os.path.exists(save_dir):
        instances = []
        for frame in os.listdir(save_dir):
            out_path = os.path.join(save_dir, frame)
            instances.append({ "filepath": out_path,
                                "country": "URY",
                            })

    else:
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"{video_name}: vÃ­deo invÃ¡lido ou erro ao carregar.")
            return False

        step = total_frames // max_frames
        selected_indices = set(i * step for i in range(max_frames))
        
        current_index = 0
        extracted = 0
        
        instances = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in selected_indices:
                out_path = os.path.join(save_dir, f"frame_{extracted:03d}.jpg")
                cv2.imwrite(out_path, frame)
                extracted += 1
                if extracted >= max_frames:
                    break
            current_index += 1
            
            instances.append({ "filepath": out_path,
                            "country": "URY",
                        })
            
        cap.release()
        
    return instances


def process_all_videos_parallel(input_folder, output_folder='temp', max_frames=20, num_workers=os.cpu_count()):
    print("Getting the videos frames")
    os.makedirs(output_folder, exist_ok=True)
    
    video_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    video_infos = [(video_path, output_folder, max_frames) for video_path in video_files]

    with Pool(num_workers) as pool:
        pool.map(extract_spaced_frames_fast, video_infos)


class BaseModel:
    def __init__(self):
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError("You must implement `_load_model`.")

    def _predict(self, video_path: str) -> str:
        raise NotImplementedError("You must implement `_predict`.")

    def predict(self, video_path: str) -> str:
        return f"'{self._predict(video_path)}'"

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


class SpeciesNetBasedModel(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.class_mapping = {
                    'domestic cattle': 'cow',
                    'no_animal': 'no_animal',
                    'white-tailed deer': 'unknown_animal',
                    'leopard cat': 'unknown_animal',
                    'bird': 'bird',
                    'european hare': 'hare',
                    'nine-banded armadillo': 'armadillo',
                    'striped skunk': 'skunk',
                    'capybara': 'capybara',
                    'wild boar': 'wild_boar',
                    'eastern gray squirrel': 'unknown_animal',
                    'greater rhea': 'bird',
                    'ocelot': 'unknown_animal',
                    'lowland tapir': 'unknown_animal',
                    'wild turkey': 'bird',
                    'red deer': 'unknown_animal',
                    'human': 'human',
                    'sika deer': 'unknown_animal',
                    'southern pig-tailed macaque': 'unknown_animal',
                    'domestic dog': 'unknown_animal',
                    'rabbit and hare family': 'hare',
                    'virginia opossum': 'unknown_animal',
                    'margay': 'margay',
                    'didelphis species': 'unknown_animal',
                    'coyote': 'unknown_animal',
                    'malagasy turtle dove': 'bird',
                    'american robin': 'bird',
                    'crab-eating fox': 'unknown_animal',
                    'domestic cat': 'unknown_animal',
                    'whiptail wallaby': 'unknown_animal',
                    'red-necked wallaby': 'unknown_animal',
                    'blank':'no_animal'
                }

    def generate_submission(self, eval_dir: str) -> None: 
        process_all_videos_parallel(eval_dir) 
        super().generate_submission(eval_dir)

    def _load_model(self) -> None:
        """
        Loads the model from a public dataset previously uploaded.
        This guarantees that the model runs OFFLINE
        """
        self.model = SpeciesNet('/kaggle/input/google-speciesnet-correct-url')

    def _predict(self, video_path: str) -> int:
        """
        Uses the loaded model to generate a prediction.
        """
    
        instances = extract_spaced_frames_fast((video_path,"temp",10))

        predictions_dict = self.model.predict(
                                        instances_dict={
                                            "instances": instances
                                        }
                                    )
                                    
        final_class = "blank"
        final_score = 0 

        for idx, pred in enumerate(predictions_dict['predictions']):

            classification = pred['classifications']

            cur_class = classification['classes'][0].split(";")[-1]
            cur_score = classification['scores'][0]
            
            if (cur_score > final_score) and (cur_class != "blank"):
                final_class = cur_class
                final_score = cur_score
        
        return self.class_mapping[final_class] if final_class in self.class_mapping else "unknown_animal"


model = SpeciesNetBasedModel()
model.generate_submission(eval_dir=EVAL_DIR)


df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv")).sort_values("Filename")
df_submission = pd.read_csv("submission.csv").sort_values("Filename")


f1_score(df_test["Species"].values, df_submission["Species"].values, average="weighted")

