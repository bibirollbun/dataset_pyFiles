# Requirements

!pip install megadetector==5.0.28
!pip install speciesnet==5.0.0
!pip install opencv-python
!pip install protobuf==3.20.*

#!pip install --no-index --find-links=/kaggle/input/cupybara-ariel-malowany-dependencies/packages megadetector --no-deps
#!pip install --no-index --find-links=/kaggle/input/cupybara-ariel-malowany-dependencies/packages speciesnet --no-deps
#!pip install --no-index --find-links=/kaggle/input/cupybara-ariel-malowany-dependencies/packages opencv-python --no-deps
#!pip install --no-index --find-links=/kaggle/input/cupybara-ariel-malowany-dependencies/packages protobuf --no-deps
#!pip install --no-index --find-links=/kaggle/input/cupybara-ariel-malowany-dependencies/packages humanfriendly --no-deps


# Import packages

import os
import sys
import ast
import kagglehub
import random
import subprocess
from datetime import datetime
import json
import joblib
import shutil
import pandas as pd
import numpy as np
import tqdm
from typing import List
import cv2
from PIL import Image
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from megadetector.detection import run_detector
from sklearn.metrics.pairwise import cosine_similarity
import torch
from speciesnet import SpeciesNet


LABELS = ["'armadillo'", "'bird'", "'capybara'", "'cow'", "'dusky_legged_guan'",
    "'gray_brocket'", "'hare'", "'human'", "'insect'", "'margay'", "'no_animal'",
    "'skunk'", "'unknown_animal'", "'wild_boar'"]


!ls /kaggle/input/cupybara/dataset/


DATASET_ROOT = "/kaggle/input/cupybara/dataset/dataset/"
TRAIN_DIR = os.path.join(DATASET_ROOT,"train")  # Directory containing train `.mp4` files
EVAL_DIR =  os.path.join(DATASET_ROOT,"test")  # Directory containing evaluation `.mp4` files
TRAIN = False


!rm -rf model_weights/ model_weights.zip submission.csv


!ls /kaggle/input/native-fauna-dummy-model


# Helper functions

def save_image(frame, file_name, append = None, save_dir = '/kaggle/working/extracted_images'):
  base_dir = os.path.join(save_dir, str(file_name))
  os.makedirs(base_dir, exist_ok = True)
  if append is not None:
    file_name = f"{file_name}_{append}"
  save_path = os.path.join(base_dir, f'{file_name}.jpg')
  cv2.imwrite(save_path, frame)

def save_json(json_file, file_name, save_dir = '/kaggle/working/extracted_images'):
  save_dir = os.path.join(save_dir, str(file_name))
  os.makedirs(save_dir, exist_ok = True)
  file_dir = os.path.join(save_dir, 'yolo_metadata.json')
  with open(file_dir, 'w') as f:
    yolo_metadata = json.dumps(json_file)
    f.write(yolo_metadata)

def open_video(file_path):
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
  else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  return cap, frame_count

def extract_frame(cap_obj, frame_number=1, save_img=False, save_path=None):
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap_obj.read()
    
    if not ret:
        print(f"No fue posible extraer el frame {frame_number}")
        return None

    if save_img and save_path is not None:
        save_image(frame, save_path)

    return frame


class BaseModel:
    def __init__(self):
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError("You must implement `_load_model`.")

    def _predict(self, video_path: str) -> str:
        raise NotImplementedError("You must implement `_predict`.")

    def predict(self, video_path: str) -> str:
        return self._predict(video_path) # Changed to return strings instead of indexes

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
        os.makedirs('/kaggle/working/model', exist_ok=True)
        species_net_files = os.listdir('/kaggle/input/cupybara-ariel-malowany-model')
        for file in species_net_files:
          shutil.copyfile(f"""/kaggle/input/cupybara-ariel-malowany-model/{file}""", f"""/kaggle/working/model/{file}""")
        self.custom_megadetector_model = run_detector.load_detector('/kaggle/working/model/md_v5a.0.0.pt')
        self.species_net = SpeciesNet('/kaggle/working/model')
        with open('/kaggle/working/model/species_dict.json', 'r') as species_dict:
          self.species_dict = json.load(species_dict)

    def _detect_image_objects(self, frame, threshold=0.2, model = None):
          if model is None:
            model = self.custom_megadetector_model
            result = model.generate_detections_one_image(frame)
            detections = result.get('detections', [])
            filtered = {str(i): {"category": d["category"], "confidence": d["conf"], "bbox": d["bbox"]}
                              for i, d in enumerate(detections) if d["conf"] > threshold}
            return filtered

    def _retrieve_prediction(self, img, frame, obj):
        try:
            predictions_dict = self.species_net.classify(
                filepaths=[f'/kaggle/working/cropped_images/{img}/{img}_{frame}_{obj}.jpg'],
                country = ['ARG', 'BRA', 'PRY', 'URY']
            )
    
            preds = predictions_dict["predictions"][0]
            if "classifications" not in preds:
                return "unknown", 0.0
    
            scores = preds["classifications"]["scores"]
            classes = preds["classifications"]["classes"]
    
            return [(x, y) for x, y in zip(classes, scores)]
    
        except Exception as e:
            print(f"Prediction failed for {img}_{frame}_{obj}: {e}")
            return "error", 0.0

    def _crop_and_save_image(self, array, detection_metadata, file_name, frame_number,
                            return_dict=False, save_dir='/kaggle/working/cropped_images', full_image = False):
        height, width = array.shape[:2]
        for idx_str, obj in detection_metadata.items():
            bbx = obj["bbox"]
            x1 = int(bbx[0] * width)
            y1 = int(bbx[1] * height)
            x2 = int((bbx[0] + bbx[2]) * width)
            y2 = int((bbx[1] + bbx[3]) * height)
    
            crop_img = array[y1:y2, x1:x2]
            append = f"{frame_number}_{idx_str}"
            if full_image:
              to_save = array
            else:
              to_save = crop_img
            save_image(to_save, file_name, append=append, save_dir=save_dir)
    
            predictions = self._retrieve_prediction(file_name, str(frame_number), idx_str)
            if return_dict:
                obj["pred_class"] = predictions
    
        if return_dict:
            return detection_metadata

    def _detect_and_predict_image(self, file_path, open_video_data = None, steps=10, find_n_frames = 3, threshold=0.5, not_indexes = None, save_dir_path='/kaggle/working/cropped_images'):
        if open_video_data is not None:
          cap_obj, frame_count = open_video_data
        else: 
          cap_obj, frame_count = open_video(file_path)
        if not_indexes is None:
            indexes = np.linspace(0, frame_count - 1, steps, dtype=int)
            iterated_frames = indexes.tolist()
        else:
            frame_seq = list(range(frame_count - 1))
            frame_seq = list(set(frame_seq) - set(not_indexes))
            frame_idx = list(np.linspace(0, len(frame_seq) - 1, steps, dtype=int))
            indexes = [frame_seq[i] for i in frame_idx]
            iterated_frames = not_indexes + indexes
        image_yolo_metadata = {}
        video_predictions = {}
        file_name = os.path.basename(file_path)
        save_file_name = file_name.replace('.mp4', '')
    
        frames_with_objects = 0
        i = 0
        category = []
        while i < len(indexes) and frames_with_objects < find_n_frames:
            frame_num = indexes[i]
            frame = extract_frame(cap_obj, frame_num)
            if frame is None:
                continue
            
            detection_metadata = self._detect_image_objects(frame, threshold = threshold)
            if detection_metadata:
                frames_with_objects += 1
                # Update detection_metadata with predictions
                detection_metadata = self._crop_and_save_image(
                    frame, detection_metadata, save_file_name, frame_num, return_dict=True, save_dir = save_dir_path
                )
                image_yolo_metadata[str(frame_num)] = detection_metadata
                for obj in list(detection_metadata.keys()):
                    obj_metadata = detection_metadata.get(obj)
                    video_pred = obj_metadata["pred_class"]
                    category.append(int(obj_metadata["category"]))
                    found_classes = video_predictions.keys()
                    for c, s in video_pred:
                      if c not in found_classes:
                          video_predictions[c] = s
                      max_score = video_predictions[c]
                      if s > max_score:
                        video_predictions[c] = s
            i +=1
        if frames_with_objects > 0:
          image_yolo_metadata["category"] = np.mean(category)
        image_yolo_metadata["frames_with_objects"] = frames_with_objects
        image_yolo_metadata["iterated_frames"] = iterated_frames
        image_yolo_metadata["video_predictions"] = dict(sorted(video_predictions.items(), key=lambda item: item[1], reverse = True))
        save_json(image_yolo_metadata, save_file_name, save_dir_path)
    
        return image_yolo_metadata

    def _species_net_to_cupybara(self, yolo_metadata, species_dict = None):
        if species_dict is None:
          species_dict = self.species_dict
        yolo_dict = yolo_metadata
        yolo_keys = yolo_dict.keys()
        video_predictions = yolo_dict["video_predictions"]
        species_list = species_dict.keys()
        mapped_predicted_species = {}
        for pred_class, score in video_predictions.items():
            for species in species_list:
                if species in pred_class:
                    label = species_dict[species]
                    if label not in mapped_predicted_species.keys() and score >= 0.05:
                        mapped_predicted_species[label] = score
                    if score >= 0.05 and mapped_predicted_species[label] < score:
                        mapped_predicted_species[label] = score
        return mapped_predicted_species

    def _final_predict(self, yolo_metadata, speciesnet_preds):
        frames_with_objects = yolo_metadata.get('frames_with_objects', 0)
        md_category = yolo_metadata.get('category')
        prediction = 'no_animal'
    
        if frames_with_objects == 0:
            return prediction 
    
        predicted_classes = list(speciesnet_preds.keys())
        unk_score = speciesnet_preds.get('unknown_animal', 0)
        no_unk_preds = [cls for cls in predicted_classes if cls != 'unknown_animal']
    
        if not no_unk_preds:
            if md_category == 1:
                return 'unknown_animal'
            elif md_category == 2:
                return 'human'
            return prediction  
    
        top_no_unk_class = no_unk_preds[0]
        top_no_unk_score = speciesnet_preds.get(top_no_unk_class, 0)
    
        all_no_unk_scores = [speciesnet_preds[cls] for cls in no_unk_preds]
        confounded_threshold = np.mean(all_no_unk_scores) + 2.5 * np.std(all_no_unk_scores)
    
        bird_like = {'dusky_legged_guan', 'bird'}
        confounded_birds = bird_like | {'squirrel'}
        birds_in_preds = [cls for cls in no_unk_preds if cls in bird_like]
        confounded_in_preds = [cls for cls in no_unk_preds if cls in confounded_birds]
    
        if len(predicted_classes) == 1:
            prediction = predicted_classes[0]
        elif top_no_unk_score > unk_score and top_no_unk_score >= 0.25:
            prediction = top_no_unk_class
        elif len(top_no_unk_class) == 1 and top_no_unk_score >= 0.10:
            prediction = top_no_unk_class
        elif set(no_unk_preds) == bird_like:
            prediction = 'dusky_legged_guan'
        elif set(no_unk_preds) == confounded_birds:
            prediction = 'bird'
        elif unk_score > top_no_unk_score:
            if top_no_unk_score > 0.50 or top_no_unk_score > confounded_threshold:
                prediction = top_no_unk_class
            else:
                prediction = 'unknown_animal'
        else:
            prediction = top_no_unk_class 
    
        if prediction in {'domestic_cat', 'fox', 'squirrel'}:
            prediction = 'unknown_animal'
    
        return prediction

    def _predict(self, video_path: str) -> int:
        """
        Uses the loaded model to generate a prediction.
        """
        yolo_metadata = self._detect_and_predict_image(video_path, open_video_data = None, steps = 20, find_n_frames = 5, threshold = 0.5)
        speciesnet_preds = self._species_net_to_cupybara(yolo_metadata)
        prediction = self._final_predict(yolo_metadata, speciesnet_preds)
        return prediction


model = CustomModel()
model.generate_submission(eval_dir=EVAL_DIR)


df_test = pd.read_csv(os.path.join(DATASET_ROOT, "test.csv")).sort_values("Filename")
df_submission = pd.read_csv("submission.csv").sort_values("Filename")


f1_score(df_test["Species"].values, df_submission["Species"].values, average="weighted")

